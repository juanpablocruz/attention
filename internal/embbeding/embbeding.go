package embbeding

import (
	"fmt"
	"log"
	"math/rand/v2"

	"github.com/juanpablocruz/attention/gen/internal/matrix"
)

type EmbeddingVec struct {
	Data []float32
}

func (v *EmbeddingVec) Add(t []float32) *EmbeddingVec {
	if len(v.Data) != len(t) {
		log.Fatal("embedding vector size mismatch")
	}

	for i := range len(v.Data) {
		v.Data[i] = v.Data[i] + t[i]
	}
	return v
}

func (v *EmbeddingVec) AddEmbbeding(t EmbeddingVec) *EmbeddingVec {
	for i := range len(v.Data) {
		v.Data[i] = v.Data[i] + t.Data[i]
	}
	return v
}

type Embedding struct {
	Vec        []EmbeddingVec
	Vocab      int
	Dimensions int
}

func (v *Embedding) Print() {
	for j := range len(v.Vec) {
		fmt.Printf("[%d]: [", j)
		for i := range len(v.Vec[j].Data) {
			fmt.Printf("%f", v.Vec[j].Data[i])
			if i < len(v.Vec[j].Data)-1 {
				fmt.Print(", ")
			}
		}
		fmt.Print("]\n")
	}
}

func (e *Embedding) GetVecForEntry(n uint8) *EmbeddingVec {
	if int(n) >= len(e.Vec) {
		log.Fatal("number out of scope")
	}

	return &e.Vec[n]
}

func (e *Embedding) SubScaledBySequence(tokens []uint8, dSequence *matrix.Matrix, scale float32) {
	if dSequence == nil {
		log.Fatal("nil sequence gradient")
	}
	if len(tokens) != dSequence.Rows {
		log.Fatal("sequence gradient rows mismatch")
	}
	if dSequence.Cols != e.Dimensions {
		log.Fatal("sequence gradient columns mismatch")
	}

	for pos := range tokens {
		token := int(tokens[pos])
		if token < 0 || token >= len(e.Vec) {
			log.Fatal("token out of embedding scope")
		}

		for j := 0; j < e.Dimensions; j++ {
			e.Vec[token].Data[j] -= scale * dSequence.Vec[pos][j]
		}
	}
}

func (e *Embedding) SubScaledByGrad(grad *matrix.Matrix, scale float32) {
	if grad == nil {
		log.Fatal("nil embedding gradient")
	}
	if grad.Rows != e.Vocab || grad.Cols != e.Dimensions {
		log.Fatal("embedding gradient shape mismatch")
	}

	for i := 0; i < e.Vocab; i++ {
		for j := 0; j < e.Dimensions; j++ {
			e.Vec[i].Data[j] -= scale * grad.Vec[i][j]
		}
	}
}

func New(vocab, dimensions int) *Embedding {
	if vocab <= 0 || dimensions <= 0 {
		log.Fatal("vocab and dimensions must be greater than 0")
	}

	vec := make([]EmbeddingVec, vocab)

	for i := range vocab {
		v := EmbeddingVec{Data: make([]float32, dimensions)}
		for j := range dimensions {
			// Use zero-centered initialization to avoid the all-positive bias from rand.Float32().
			// This reduces immediate single-class collapse in argmax, but it does not replace training.
			v.Data[j] = rand.Float32()*0.2 - 0.1
		}
		vec[i] = v
	}

	return &Embedding{Vec: vec, Vocab: vocab, Dimensions: dimensions}
}
