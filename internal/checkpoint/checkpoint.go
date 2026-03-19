package checkpoint

import (
	"encoding/gob"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/juanpablocruz/attention/gen/internal/embbeding"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/transformer"
)

type matrixState struct {
	Rows int
	Cols int
	Data []float32
}

type embeddingState struct {
	Vocab      int
	Dimensions int
	Data       []float32
}

type modelState struct {
	Version   int
	Embedding embeddingState
	WQ        matrixState
	WK        matrixState
	WV        matrixState
	W1        matrixState
	W2        matrixState
	CW        matrixState
}

func Save(path string, e *embbeding.Embedding, t *transformer.TransformerBlock, cW *matrix.Matrix) error {
	state := modelState{
		Version:   1,
		Embedding: fromEmbedding(e),
		WQ:        fromMatrix(t.WQ),
		WK:        fromMatrix(t.WK),
		WV:        fromMatrix(t.WV),
		W1:        fromMatrix(t.W1),
		W2:        fromMatrix(t.W2),
		CW:        fromMatrix(cW),
	}

	dir := filepath.Dir(path)
	if dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	return enc.Encode(state)
}

func Load(path string, e *embbeding.Embedding, t *transformer.TransformerBlock, cW *matrix.Matrix) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	defer f.Close()

	dec := gob.NewDecoder(f)
	var state modelState
	if err := dec.Decode(&state); err != nil {
		return false, err
	}

	if state.Version != 1 {
		return false, fmt.Errorf("unsupported checkpoint version: %d", state.Version)
	}

	if err := toEmbedding(state.Embedding, e); err != nil {
		return false, err
	}
	if err := toMatrix(state.WQ, t.WQ); err != nil {
		return false, err
	}
	if err := toMatrix(state.WK, t.WK); err != nil {
		return false, err
	}
	if err := toMatrix(state.WV, t.WV); err != nil {
		return false, err
	}
	if err := toMatrix(state.W1, t.W1); err != nil {
		return false, err
	}
	if err := toMatrix(state.W2, t.W2); err != nil {
		return false, err
	}
	if err := toMatrix(state.CW, cW); err != nil {
		return false, err
	}

	return true, nil
}

func fromMatrix(m *matrix.Matrix) matrixState {
	out := matrixState{Rows: m.Rows, Cols: m.Cols, Data: make([]float32, 0, m.Rows*m.Cols)}
	for i := 0; i < m.Rows; i++ {
		out.Data = append(out.Data, m.Vec[i]...)
	}
	return out
}

func toMatrix(state matrixState, m *matrix.Matrix) error {
	if state.Rows != m.Rows || state.Cols != m.Cols {
		return fmt.Errorf("matrix shape mismatch: checkpoint=%dx%d current=%dx%d", state.Rows, state.Cols, m.Rows, m.Cols)
	}
	if len(state.Data) != m.Rows*m.Cols {
		return fmt.Errorf("matrix data size mismatch")
	}

	idx := 0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Vec[i][j] = state.Data[idx]
			idx++
		}
	}
	return nil
}

func fromEmbedding(e *embbeding.Embedding) embeddingState {
	out := embeddingState{Vocab: e.Vocab, Dimensions: e.Dimensions, Data: make([]float32, 0, e.Vocab*e.Dimensions)}
	for i := 0; i < e.Vocab; i++ {
		out.Data = append(out.Data, e.Vec[i].Data...)
	}
	return out
}

func toEmbedding(state embeddingState, e *embbeding.Embedding) error {
	if state.Vocab != e.Vocab || state.Dimensions != e.Dimensions {
		return fmt.Errorf("embedding shape mismatch: checkpoint=%dx%d current=%dx%d", state.Vocab, state.Dimensions, e.Vocab, e.Dimensions)
	}
	if len(state.Data) != e.Vocab*e.Dimensions {
		return fmt.Errorf("embedding data size mismatch")
	}

	idx := 0
	for i := 0; i < e.Vocab; i++ {
		for j := 0; j < e.Dimensions; j++ {
			e.Vec[i].Data[j] = state.Data[idx]
			idx++
		}
	}
	return nil
}
