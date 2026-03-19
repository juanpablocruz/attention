package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/juanpablocruz/attention/gen/internal/checkpoint"
	"github.com/juanpablocruz/attention/gen/internal/embbeding"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/prompt"
	"github.com/juanpablocruz/attention/gen/internal/transformer"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: predict \"sort list [3, 4,2,5,1] desc\"")
	}

	inputPrompt := strings.Join(os.Args[1:], " ")
	source, err := prompt.Encode(inputPrompt)
	if err != nil {
		log.Fatalf("invalid prompt: %v", err)
	}

	numbers, order, err := prompt.Parse(inputPrompt)
	if err != nil {
		log.Fatalf("invalid prompt: %v", err)
	}
	targetPrompt := prompt.BuildTarget(numbers, order)
	targetTokens, err := prompt.Encode(targetPrompt)
	if err != nil {
		log.Fatalf("could not encode target prompt: %v", err)
	}

	const (
		vocabSize      = prompt.VocabSize
		modelDim       = 32
		sequenceLen    = prompt.SequenceLen
		checkpointPath = "./checkpoints/embed_model.bin"
	)

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

	fmt.Printf("prompt:    %s\n", inputPrompt)
	fmt.Printf("predicted: %v\n", pred[2:7])
	fmt.Printf("expected:  %v\n", targetTokens[2:7])
}
