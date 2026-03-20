package transformer

import (
	"log"

	"github.com/juanpablocruz/attention/gen/internal/matrix"
)

type TransformerBlock struct {
	NumHeads int
	HeadDim  int

	WQ, WK, WV []*matrix.Matrix // Per-head attention weights
	WO         *matrix.Matrix   // Output projection after concat
	W1, W2     *matrix.Matrix   // FFN weights
}

func New(dimensions, numHeads int) *TransformerBlock {
	if numHeads <= 0 {
		log.Fatal("numHeads must be > 0")
	}
	if dimensions%numHeads != 0 {
		log.Fatalf("dimensions (%d) must be divisible by numHeads (%d)", dimensions, numHeads)
	}

	expandedSize := dimensions * 4
	t := matrix.New(dimensions, expandedSize)
	w := matrix.New(expandedSize, dimensions)

	headDim := dimensions / numHeads
	wQ := make([]*matrix.Matrix, numHeads)
	wK := make([]*matrix.Matrix, numHeads)
	wV := make([]*matrix.Matrix, numHeads)
	for h := range numHeads {
		wQ[h] = matrix.New(dimensions, headDim)
		wK[h] = matrix.New(dimensions, headDim)
		wV[h] = matrix.New(dimensions, headDim)
	}
	wO := matrix.New(dimensions, dimensions)

	return &TransformerBlock{
		NumHeads: numHeads,
		HeadDim:  headDim,
		WQ:       wQ,
		WK:       wK,
		WV:       wV,
		WO:       wO,
		W1:       t,
		W2:       w,
	}
}
