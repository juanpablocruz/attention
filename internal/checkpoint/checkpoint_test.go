package checkpoint

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/juanpablocruz/attention/gen/internal/embbeding"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/transformer"
)

func TestLoadMissingFile(t *testing.T) {
	t.Parallel()

	e := embbeding.New(8, 4)
	tb := transformer.New(4, 2)
	cw := matrix.NewZeroMatrix(4, 8)

	ok, err := Load(filepath.Join(t.TempDir(), "missing.gob"), e, tb, cw)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if ok {
		t.Fatal("Load() ok = true, want false")
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "checkpoint.gob")

	wantEmbedding := embbeding.New(8, 4)
	wantBlock := transformer.New(4, 2)
	wantCW := matrix.NewZeroMatrix(4, 8)
	fillEmbedding(wantEmbedding, 1)
	fillTransformer(wantBlock, 100)
	fillMatrix(wantCW, 200)

	if err := Save(path, wantEmbedding, wantBlock, wantCW); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	gotEmbedding := embbeding.New(8, 4)
	gotBlock := transformer.New(4, 2)
	gotCW := matrix.NewZeroMatrix(4, 8)

	ok, err := Load(path, gotEmbedding, gotBlock, gotCW)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if !ok {
		t.Fatal("Load() ok = false, want true")
	}

	if !reflect.DeepEqual(gotEmbedding, wantEmbedding) {
		t.Fatal("loaded embedding does not match saved embedding")
	}
	if !reflect.DeepEqual(gotCW, wantCW) {
		t.Fatal("loaded classifier weights do not match saved weights")
	}
	if !reflect.DeepEqual(gotBlock, wantBlock) {
		t.Fatal("loaded transformer block does not match saved block")
	}
}

func TestLoadRejectsHeadMismatch(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "checkpoint.gob")

	srcEmbedding := embbeding.New(8, 4)
	srcBlock := transformer.New(4, 2)
	srcCW := matrix.NewZeroMatrix(4, 8)
	fillEmbedding(srcEmbedding, 1)
	fillTransformer(srcBlock, 10)
	fillMatrix(srcCW, 20)

	if err := Save(path, srcEmbedding, srcBlock, srcCW); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	dstEmbedding := embbeding.New(8, 4)
	dstBlock := transformer.New(4, 1)
	dstCW := matrix.NewZeroMatrix(4, 8)

	ok, err := Load(path, dstEmbedding, dstBlock, dstCW)
	if err == nil {
		t.Fatal("Load() error = nil, want error")
	}
	if ok {
		t.Fatal("Load() ok = true, want false")
	}
}

func fillEmbedding(e *embbeding.Embedding, start float32) {
	value := start
	for i := range e.Vec {
		for j := range e.Vec[i].Data {
			e.Vec[i].Data[j] = value
			value += 1
		}
	}
}

func fillTransformer(tb *transformer.TransformerBlock, start float32) {
	value := start
	for _, m := range tb.WQ {
		value = fillMatrix(m, value)
	}
	for _, m := range tb.WK {
		value = fillMatrix(m, value)
	}
	for _, m := range tb.WV {
		value = fillMatrix(m, value)
	}
	value = fillMatrix(tb.WO, value)
	value = fillMatrix(tb.W1, value)
	_ = fillMatrix(tb.W2, value)
}

func fillMatrix(m *matrix.Matrix, start float32) float32 {
	value := start
	for i := range m.Vec {
		for j := range m.Vec[i] {
			m.Vec[i][j] = value
			value += 1
		}
	}
	return value
}
