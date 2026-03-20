package matrix

func mulGeneric(a, b *Matrix) *Matrix {
	result := NewZeroMatrix(a.Rows, b.Cols)

	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			var sum float32
			for k := 0; k < a.Cols; k++ {
				sum += a.Vec[i][k] * b.Vec[k][j]
			}
			result.Vec[i][j] = sum
		}
	}

	return result
}
