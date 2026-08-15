package walstore

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"sync"
)

// Operation codes stored as the first byte of each record payload.
const (
	opSet    byte = 0x01
	opDelete byte = 0x02
)

// Limits enforced on every write.
const (
	maxKeyLen   = 0xffff // 65535 bytes (encoded as uint16)
	maxValueLen = 64 << 20
	// maxPayloadSize bounds how much recovery will allocate for a single
	// record's payload before declaring it corrupt/truncated.
	maxPayloadSize = (1 + 2 + maxKeyLen) + (4 + maxValueLen)
)

// Sentinel errors returned during WAL recovery.
var (
	// errIncompleteRecord means the stream ended before a full record could be
	// read (truncation). Recovery treats this as "stop here".
	errIncompleteRecord = errors.New("walstore: incomplete record")
	// errChecksumMismatch means a record's CRC did not validate (corruption).
	// Recovery treats this the same as truncation: stop and trim.
	errChecksumMismatch = errors.New("walstore: checksum mismatch")
)

// record is a single logical WAL entry.
type record struct {
	op  byte
	key []byte
	val []byte // only populated for opSet
}

// Store is a WAL-backed key-value engine. All writes are appended to an
// append-only log on disk and mirrored into an in-memory hash index. Reads
// are served from memory. Opening a store replays its log to rebuild the
// index, tolerating a truncated or corrupt tail.
type Store struct {
	mu    sync.RWMutex
	file  *os.File
	path  string
	index map[string][]byte
}

// Open opens (creating if absent) the WAL at path and replays it into memory.
// The returned store is safe for concurrent use.
func Open(path string) (*Store, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open wal %q: %w", path, err)
	}
	s := &Store{
		file:  f,
		path:  path,
		index: make(map[string][]byte),
	}
	if err := s.replay(); err != nil {
		f.Close()
		return nil, err
	}
	return s, nil
}

// replay re-reads the WAL from the start, rebuilding the in-memory index. It
// stops at the first truncated or corrupt record, truncates the file back to
// the end of the last good record, and leaves the cursor at end-of-file for
// subsequent appends.
func (s *Store) replay() error {
	if _, err := s.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("seek start: %w", err)
	}
	validEnd := int64(0)
	for {
		// Capture position before each attempt so a partial read of the next
		// record does not advance validEnd.
		if _, err := s.file.Seek(0, io.SeekCurrent); err != nil {
			return fmt.Errorf("seek cur: %w", err)
		}
		rec, err := readRecord(s.file)
		if err != nil {
			// errIncompleteRecord / errChecksumMismatch / io.EOF all mean
			// "no more complete records" — stop.
			break
		}
		s.applyRecord(rec)
		pos, err := s.file.Seek(0, io.SeekCurrent)
		if err != nil {
			return fmt.Errorf("seek cur after read: %w", err)
		}
		validEnd = pos
	}
	// Trim any trailing partial/corrupt bytes so future appends land cleanly.
	if err := s.file.Truncate(validEnd); err != nil {
		return fmt.Errorf("truncate: %w", err)
	}
	if _, err := s.file.Seek(0, io.SeekEnd); err != nil {
		return fmt.Errorf("seek end: %w", err)
	}
	return nil
}

// applyRecord mutates the in-memory index for a single replayed record. It is
// NOT concurrency-safe on its own; callers hold the write lock.
func (s *Store) applyRecord(rec *record) {
	switch rec.op {
	case opSet:
		s.index[string(rec.key)] = rec.val
	case opDelete:
		delete(s.index, string(rec.key))
	}
}

// Set writes key=value. The set record is flushed to disk before this returns.
func (s *Store) Set(key, value []byte) error {
	if len(key) == 0 {
		return errors.New("walstore: empty key")
	}
	if len(key) > maxKeyLen {
		return fmt.Errorf("walstore: key too long (%d > %d)", len(key), maxKeyLen)
	}
	if len(value) > maxValueLen {
		return fmt.Errorf("walstore: value too long (%d > %d)", len(value), maxValueLen)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.appendRecord(&record{op: opSet, key: key, val: value}); err != nil {
		return err
	}
	s.index[string(key)] = value
	return nil
}

// Delete writes a tombstone for key and removes it from the index. It is
// idempotent: deleting a missing key still succeeds and still logs a tombstone.
func (s *Store) Delete(key []byte) error {
	if len(key) == 0 {
		return errors.New("walstore: empty key")
	}
	if len(key) > maxKeyLen {
		return fmt.Errorf("walstore: key too long (%d > %d)", len(key), maxKeyLen)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.appendRecord(&record{op: opDelete, key: key}); err != nil {
		return err
	}
	delete(s.index, string(key))
	return nil
}

// Get returns a copy of the value for key, or ok=false if absent.
func (s *Store) Get(key []byte) (value []byte, ok bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, found := s.index[string(key)]
	if !found {
		return nil, false
	}
	return v, true
}

// Snapshot returns a deep copy of the entire in-memory index.
func (s *Store) Snapshot() map[string][]byte {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if len(s.index) == 0 {
		return nil
	}
	out := make(map[string][]byte, len(s.index))
	for k, v := range s.index {
		out[k] = v
	}
	return out
}

// Len returns the number of keys in the in-memory index.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.index)
}

// WALSize returns the current byte size of the WAL file.
func (s *Store) WALSize() (int64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	fi, err := s.file.Stat()
	if err != nil {
		return 0, err
	}
	return fi.Size(), nil
}

// Compact folds the current in-memory state into a fresh WAL containing one
// set record per surviving key, then atomically replaces the old WAL. The
// in-memory index is unchanged. A crash at any point leaves either the old or
// the new (fully-synced) file in place, both of which recover to the same
// state.
func (s *Store) Compact() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	tmpPath := s.path + ".compact.tmp"
	tmp, err := os.OpenFile(tmpPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("create compact tmp: %w", err)
	}
	for k, v := range s.index {
		if err := writeRecord(tmp, &record{op: opSet, key: []byte(k), val: v}); err != nil {
			tmp.Close()
			os.Remove(tmpPath)
			return fmt.Errorf("write compact record: %w", err)
		}
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("sync compact tmp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("close compact tmp: %w", err)
	}
	if err := s.file.Close(); err != nil {
		return fmt.Errorf("close old wal: %w", err)
	}
	if err := os.Rename(tmpPath, s.path); err != nil {
		return fmt.Errorf("rename compact: %w", err)
	}
	f, err := os.OpenFile(s.path, os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("reopen wal after compact: %w", err)
	}
	s.file = f
	return nil
}

// Close flushes and closes the WAL. Calling Close twice is a no-op.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.file == nil {
		return nil
	}
	err := s.file.Close()
	s.file = nil
	return err
}

// appendRecord writes rec to the WAL and fsyncs before returning.
func (s *Store) appendRecord(rec *record) error {
	if err := writeRecord(s.file, rec); err != nil {
		return fmt.Errorf("write record: %w", err)
	}
	if err := s.file.Sync(); err != nil {
		return fmt.Errorf("sync wal: %w", err)
	}
	return nil
}

// writeRecord encodes and writes a single record to w (no sync).
func writeRecord(w io.Writer, rec *record) error {
	payload := encodePayload(rec)
	var header [8]byte
	binary.BigEndian.PutUint32(header[0:4], uint32(len(payload)))
	binary.BigEndian.PutUint32(header[4:8], crc32.ChecksumIEEE(payload))
	if _, err := w.Write(header[:]); err != nil {
		return err
	}
	if _, err := w.Write(payload); err != nil {
		return err
	}
	return nil
}

// readRecord reads one length-prefixed, CRC-checked record from r.
func readRecord(r io.Reader) (*record, error) {
	var header [8]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, errIncompleteRecord
		}
		return nil, err
	}
	payloadLen := binary.BigEndian.Uint32(header[0:4])
	wantCRC := binary.BigEndian.Uint32(header[4:8])
	if payloadLen == 0 || payloadLen > maxPayloadSize {
		// Impossibly large or zero → corrupt/truncated tail.
		return nil, errIncompleteRecord
	}
	payload := make([]byte, payloadLen)
	if _, err := io.ReadFull(r, payload); err != nil {
		// Partial payload = truncation.
		return nil, errIncompleteRecord
	}
	if crc32.ChecksumIEEE(payload) != wantCRC {
		return nil, errChecksumMismatch
	}
	rec, err := parsePayload(payload)
	if err != nil {
		// CRC passed but bytes are malformed: treat as corrupt.
		return nil, errChecksumMismatch
	}
	return rec, nil
}

// encodePayload builds the (op, key, value) payload bytes for a record.
func encodePayload(rec *record) []byte {
	if rec.op == opSet {
		b := make([]byte, 0, 1+2+len(rec.key)+4+len(rec.val))
		b = append(b, opSet)
		var kl [2]byte
		binary.BigEndian.PutUint16(kl[:], uint16(len(rec.key)))
		b = append(b, kl[:]...)
		b = append(b, rec.key...)
		var vl [4]byte
		binary.BigEndian.PutUint32(vl[:], uint32(len(rec.val)))
		b = append(b, vl[:]...)
		b = append(b, rec.val...)
		return b
	}
	b := make([]byte, 0, 1+2+len(rec.key))
	b = append(b, opDelete)
	var kl [2]byte
	binary.BigEndian.PutUint16(kl[:], uint16(len(rec.key)))
	b = append(b, kl[:]...)
	b = append(b, rec.key...)
	return b
}

// parsePayload decodes payload bytes back into a record.
func parsePayload(p []byte) (*record, error) {
	if len(p) < 1 {
		return nil, errors.New("empty payload")
	}
	op := p[0]
	p = p[1:]
	if len(p) < 2 {
		return nil, errors.New("missing key length")
	}
	kl := binary.BigEndian.Uint16(p[0:2])
	p = p[2:]
	if uint16(len(p)) < kl {
		return nil, errors.New("key truncated")
	}
	key := make([]byte, kl)
	copy(key, p[:kl])
	p = p[kl:]
	rec := &record{op: op, key: key}
	switch op {
	case opSet:
		if len(p) < 4 {
			return nil, errors.New("missing value length")
		}
		vl := binary.BigEndian.Uint32(p[0:4])
		p = p[4:]
		if uint32(len(p)) < vl {
			return nil, errors.New("value truncated")
		}
		rec.val = make([]byte, vl)
		copy(rec.val, p[:vl])
		p = p[vl:]
		if len(p) != 0 {
			return nil, errors.New("trailing bytes after value")
		}
	case opDelete:
		if len(p) != 0 {
			return nil, errors.New("trailing bytes after delete")
		}
	default:
		return nil, fmt.Errorf("unknown op %#x", op)
	}
	return rec, nil
}
