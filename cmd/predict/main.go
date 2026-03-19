package main

import (
	"fmt"
	"log"
	"os"
	"slices"
	"strconv"

	"github.com/juanpablocruz/attention/gen/internal/checkpoint"
	"github.com/juanpablocruz/attention/gen/internal/embbeding"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/transformer"
)

func main() {
	if len(os.Args) != 6 {
		log.Fatal("usage: predict <n1> <n2> <n3> <n4> <n5>")
	}

	const (
		vocabSize      = 10
		modelDim       = 32
		sequenceLen    = 5
		checkpointPath = "./checkpoints/embed_model.bin"
	)

	var source [sequenceLen]uint8
	for i := 0; i < sequenceLen; i++ {
		n, err := strconv.Atoi(os.Args[i+1])
		if err != nil || n < 0 || n >= vocabSize {
			log.Fatalf("invalid token at position %d: %q (expected 0-%d)", i, os.Args[i+1], vocabSize-1)
		}
		source[i] = uint8(n)
	}

	m := embbeding.New(vocabSize, modelDim)
	tBlock := transformer.New(modelDim)
	cW := matrix.New(modelDim, vocabSize)

	loaded, err := checkpoint.Load(checkpointPath, m, tBlock, cW)
	if err != nil {
		log.Fatalf("could not load checkpoint: %v", err)
	}
	if !loaded {
		log.Fatalf("checkpoint not found at %s", checkpointPath)
	}

	sequenceMatrix := matrix.NewZeroMatrix(sequenceLen, modelDim)
	for j := 0; j < sequenceLen; j++ {
		vec := m.GetVecForEntry(source[j])
		positioned := transformer.AddPosition(j, vec)
		copy(sequenceMatrix.Vec[j], positioned.Data)
	}

	q := sequenceMatrix.Mul(tBlock.WQ)
	k := sequenceMatrix.Mul(tBlock.WK)
	v := sequenceMatrix.Mul(tBlock.WV)

	attention := transformer.ScaledAttention(q, k, v)
	out := transformer.AddAndNorm(sequenceMatrix, attention)
	ffn, _ := transformer.ExpandWithCache(out, tBlock.W1, tBlock.W2)

	logits := ffn.Mul(cW)
	pred := transformer.SelectChoice(logits)

	target := make([]uint8, sequenceLen)
	copy(target, source[:])
	slices.Sort(target)

	fmt.Printf("input:    %v\n", source)
	fmt.Printf("predicted:%v\n", pred)
	fmt.Printf("target:   %v\n", target)
}
