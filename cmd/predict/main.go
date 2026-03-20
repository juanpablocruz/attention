package main

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/juanpablocruz/attention/gen/internal/checkpoint"
	"github.com/juanpablocruz/attention/gen/internal/embbeding"
	"github.com/juanpablocruz/attention/gen/internal/intent"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/prompt"
	"github.com/juanpablocruz/attention/gen/internal/transformer"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: predict \"sort list [3, 4,2,5,1] desc\"")
	}

	inputPrompt := strings.Join(os.Args[1:], " ")
	numbers, err := prompt.ExtractNumbers(inputPrompt)
	if err != nil {
		log.Fatalf("invalid prompt: %v", err)
	}

	intentModel, err := intent.Load("./checkpoints/intent_model.bin")
	if err != nil {
		log.Fatalf("could not load intent model: %v", err)
	}

	label, probs := intentModel.Predict(inputPrompt)
	task, order, err := intent.LabelToTaskOrder(label)
	if err != nil {
		log.Fatalf("could not infer intent: %v", err)
	}

	source, err := prompt.EncodeStructured(task, order, numbers)
	if err != nil {
		log.Fatalf("could not encode prompt: %v", err)
	}

	const (
		vocabSize       = prompt.VocabSize
		defaultModelDim = 64
		defaultHeads    = 4
		sequenceLen     = prompt.SequenceLen
		checkpointPath  = "./checkpoints/embed_model.bin"
	)

	modelDim := defaultModelDim
	if v := os.Getenv("ATTN_MODEL_DIM"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err == nil && parsed > 0 {
			modelDim = parsed
		}
	}

	numHeads := defaultHeads
	if v := os.Getenv("ATTN_NUM_HEADS"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err == nil && parsed > 0 {
			numHeads = parsed
		}
	}

	m := embbeding.New(vocabSize, modelDim)
	tBlock := transformer.New(modelDim, numHeads)
	cW := matrix.New(modelDim, vocabSize)

	loaded, err := checkpoint.Load(checkpointPath, m, tBlock, cW)
	if err != nil {
		log.Fatalf("could not load checkpoint: %v", err)
	}
	if !loaded {
		log.Fatalf("checkpoint not found at %s", checkpointPath)
	}

	sequenceMatrix := matrix.NewZeroMatrix(sequenceLen, modelDim)
	for j := range sequenceLen {
		vec := m.GetVecForEntry(source[j])
		positioned := transformer.AddPosition(j, vec)
		copy(sequenceMatrix.Vec[j], positioned.Data)
	}

	attention, _, err := transformer.MultiHeadAttention(sequenceMatrix, tBlock)
	if err != nil {
		log.Fatalf("attention forward failed: %v", err)
	}
	out := transformer.AddAndNorm(sequenceMatrix, attention)
	ffn, _ := transformer.ExpandWithCache(out, tBlock.W1, tBlock.W2)

	logits := ffn.Mul(cW)
	pred := transformer.SelectChoice(logits)

	fmt.Printf("prompt:    %s\n", inputPrompt)
	fmt.Printf("intent:    %s (p=[asc:%.2f desc:%.2f sum:%.2f])\n", taskLabel(task, order), probs[0], probs[1], probs[2])
	if task == prompt.TaskSum {
		sumIdxA := 2 + prompt.MaxListLen - 2
		sumIdxB := 2 + prompt.MaxListLen - 1
		fmt.Printf("output:    [%d %d]\n", pred[sumIdxA], pred[sumIdxB])
		return
	}

	start := 2
	end := start + len(numbers)
	fmt.Printf("output:    %v\n", pred[start:end])
}

func taskLabel(task, order string) string {
	if task == prompt.TaskSort {
		return task + "_" + order
	}
	return task
}
