package attention

import (
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/transformer"
)

type PackedModel struct {
	Model           *Model
	Block           *transformer.PackedBlock
	Output          *matrix.PackedMatrix
	OutputTranspose *matrix.PackedMatrix
}

func (m *Model) Pack() *PackedModel {
	if m == nil {
		return nil
	}
	return &PackedModel{
		Model:           m,
		Block:           transformer.PackBlock(m.Block),
		Output:          m.Output.PackB(),
		OutputTranspose: m.Output.Transpose().PackB(),
	}
}
