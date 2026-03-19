package transformer

import "github.com/juanpablocruz/attention/gen/internal/matrix"

type TransformerBlock struct {
	WQ, WK, WV *matrix.Matrix // Attention weights
	W1, W2     *matrix.Matrix // FFN weights
}

func New(dimensions int) *TransformerBlock {
	expandedSize := dimensions * 4
	t := matrix.New(dimensions, expandedSize)
	w := matrix.New(expandedSize, dimensions)

	wQ := matrix.New(dimensions, dimensions)
	wK := matrix.New(dimensions, dimensions)
	wV := matrix.New(dimensions, dimensions)

	return &TransformerBlock{
		WQ: wQ,
		WK: wK,
		WV: wV,
		W1: t,
		W2: w,
	}
}
