package matrix

// NeonMatrix is a contiguous representation intended for SIMD-oriented kernels.
// It currently backs the arm64-friendly multiplication path used by Matrix.Mul.
type NeonMatrix struct {
	Rows int
	Cols int
	Data []float32
}

func NewNeonFromMatrix(m *Matrix) *NeonMatrix {
	if m == nil {
		return nil
	}
	out := &NeonMatrix{Rows: m.Rows, Cols: m.Cols, Data: make([]float32, m.Rows*m.Cols)}
	idx := 0
	for i := 0; i < m.Rows; i++ {
		copy(out.Data[idx:idx+m.Cols], m.Vec[i])
		idx += m.Cols
	}
	return out
}

func (n *NeonMatrix) ToMatrix() *Matrix {
	if n == nil {
		return nil
	}
	out := NewZeroMatrix(n.Rows, n.Cols)
	idx := 0
	for i := 0; i < n.Rows; i++ {
		copy(out.Vec[i], n.Data[idx:idx+n.Cols])
		idx += n.Cols
	}
	return out
}
