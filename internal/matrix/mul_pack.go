package matrix

type PackedMatrix struct {
	Data     []float32
	Original *Matrix
	FullCols int
}

func packB4(dst, src []float32, rows, cols int) {
	fullCols := cols &^ 3
	di := 0
	for j := 0; j < fullCols; j += 4 {
		for k := range rows {
			si := k*cols + j
			dst[di+0] = src[si+0]
			dst[di+1] = src[si+1]
			dst[di+2] = src[si+2]
			dst[di+3] = src[si+3]
			di += 4
		}
	}
}

// PackB packs wm for reuse as the right-hand-side matrix in repeated multiplies.
func (wm *Matrix) PackB() *PackedMatrix {
	if wm == nil {
		return nil
	}
	fullCols := wm.Cols &^ 3
	if len(wm.Data) != wm.Rows*wm.Cols || fullCols == 0 {
		return &PackedMatrix{Original: wm, FullCols: fullCols}
	}
	packed := make([]float32, wm.Rows*fullCols)
	packB4(packed, wm.Data, wm.Rows, wm.Cols)
	return &PackedMatrix{
		Data:     packed,
		Original: wm,
		FullCols: fullCols,
	}
}
