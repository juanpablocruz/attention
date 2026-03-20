package transformer

import "github.com/juanpablocruz/attention/gen/internal/matrix"

type PackedBlock struct {
	Block *TransformerBlock

	WQ, WK, WV      []*matrix.PackedMatrix
	WQTranspose     []*matrix.PackedMatrix
	WKTranspose     []*matrix.PackedMatrix
	WVTranspose     []*matrix.PackedMatrix
	WO, WOTranspose *matrix.PackedMatrix
	W1, W1Transpose *matrix.PackedMatrix
	W2, W2Transpose *matrix.PackedMatrix
}

func PackBlock(block *TransformerBlock) *PackedBlock {
	if block == nil {
		return nil
	}

	packed := &PackedBlock{
		Block:       block,
		WQ:          make([]*matrix.PackedMatrix, block.NumHeads),
		WK:          make([]*matrix.PackedMatrix, block.NumHeads),
		WV:          make([]*matrix.PackedMatrix, block.NumHeads),
		WQTranspose: make([]*matrix.PackedMatrix, block.NumHeads),
		WKTranspose: make([]*matrix.PackedMatrix, block.NumHeads),
		WVTranspose: make([]*matrix.PackedMatrix, block.NumHeads),
		WO:          block.WO.PackB(),
		WOTranspose: block.WO.Transpose().PackB(),
		W1:          block.W1.PackB(),
		W1Transpose: block.W1.Transpose().PackB(),
		W2:          block.W2.PackB(),
		W2Transpose: block.W2.Transpose().PackB(),
	}

	for h := 0; h < block.NumHeads; h++ {
		packed.WQ[h] = block.WQ[h].PackB()
		packed.WK[h] = block.WK[h].PackB()
		packed.WV[h] = block.WV[h].PackB()
		packed.WQTranspose[h] = block.WQ[h].Transpose().PackB()
		packed.WKTranspose[h] = block.WK[h].Transpose().PackB()
		packed.WVTranspose[h] = block.WV[h].Transpose().PackB()
	}

	return packed
}
