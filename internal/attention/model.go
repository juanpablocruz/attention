package attention

import (
	"fmt"

	"github.com/juanpablocruz/attention/gen/internal/checkpoint"
	"github.com/juanpablocruz/attention/gen/internal/embbeding"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/transformer"
)

// Model holds the learned components of the generator stack.
type Model struct {
	Embedding *embbeding.Embedding
	Block     *transformer.TransformerBlock
	Output    *matrix.Matrix
}

// ForwardPass captures the generator forward-pass intermediates.
type ForwardPass struct {
	Sequence  *matrix.Matrix
	Attention *matrix.Matrix
	MHACache  *transformer.MHACache
	Out       *matrix.Matrix
	FFN       *matrix.Matrix
	FFNCache  *transformer.FFNCache
	Logits    *matrix.Matrix
	Choices   []int
}

// NewModel constructs a randomly initialized generator model.
func NewModel(vocabSize, modelDim, numHeads int) *Model {
	return &Model{
		Embedding: embbeding.New(vocabSize, modelDim),
		Block:     transformer.New(modelDim, numHeads),
		Output:    matrix.New(modelDim, vocabSize),
	}
}

// LoadCheckpoint loads a saved generator checkpoint into the model.
func (m *Model) LoadCheckpoint(path string) (bool, error) {
	if m == nil {
		return false, fmt.Errorf("nil model")
	}
	return checkpoint.Load(path, m.Embedding, m.Block, m.Output)
}

// SaveCheckpoint saves the generator checkpoint to disk.
func (m *Model) SaveCheckpoint(path string) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}
	return checkpoint.Save(path, m.Embedding, m.Block, m.Output)
}

// Forward runs the embedding, attention, and projection stack over structured tokens.
func (m *Model) Forward(tokens []uint8) (*ForwardPass, error) {
	return m.ForwardPacked(tokens, nil)
}

func (m *Model) ForwardPacked(tokens []uint8, packed *PackedModel) (*ForwardPass, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token sequence")
	}

	sequence := matrix.NewZeroMatrix(len(tokens), m.Embedding.Dimensions)
	for pos := range len(tokens) {
		vec := m.Embedding.GetVecForEntry(tokens[pos])
		positioned := transformer.AddPosition(pos, vec)
		copy(sequence.Vec[pos], positioned.Data)
	}

	var packedBlock *transformer.PackedBlock
	if packed != nil {
		packedBlock = packed.Block
	}
	attentionOut, mhaCache, err := transformer.MultiHeadAttentionPacked(sequence, m.Block, packedBlock)
	if err != nil {
		return nil, err
	}

	out := transformer.AddAndNorm(sequence, attentionOut)
	var packedW1, packedW2 *matrix.PackedMatrix
	if packedBlock != nil {
		packedW1 = packedBlock.W1
		packedW2 = packedBlock.W2
	}
	ffn, ffnCache := transformer.ExpandWithCachePacked(out, m.Block.W1, m.Block.W2, packedW1, packedW2)
	var logits *matrix.Matrix
	if packed != nil && packed.Output != nil {
		logits = ffn.MulPackedInto(nil, packed.Output)
	} else {
		logits = ffn.Mul(m.Output)
	}

	return &ForwardPass{
		Sequence:  sequence,
		Attention: attentionOut,
		MHACache:  mhaCache,
		Out:       out,
		FFN:       ffn,
		FFNCache:  ffnCache,
		Logits:    logits,
		Choices:   transformer.SelectChoice(logits),
	}, nil
}
