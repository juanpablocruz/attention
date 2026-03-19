package matrix

import (
	"math"
	"math/rand/v2"
)

type Matrix struct {
	Vec  [][]float32
	Cols int
	Rows int
}

func New(rows, cols int) *Matrix {
	vec := make([][]float32, rows)
	for i := range vec {
		vec[i] = make([]float32, cols)
		for j := range vec[i] {
			vec[i][j] = rand.Float32()*0.2 - 0.1
		}
	}
	return &Matrix{Vec: vec, Rows: rows, Cols: cols}
}

func NewZeroMatrix(rows, cols int) *Matrix {
	vec := make([][]float32, rows)
	for i := range vec {
		vec[i] = make([]float32, cols)
		for j := range vec[i] {
			vec[i][j] = 0
		}
	}
	return &Matrix{Vec: vec, Rows: rows, Cols: cols}
}

func Softmax(wm *Matrix) *Matrix {
	if wm == nil || wm.Rows == 0 || wm.Cols == 0 {
		return wm
	}

	for i := range wm.Rows {
		maxVal := float64(wm.Vec[i][0])
		for j := 1; j < wm.Cols; j++ {
			if float64(wm.Vec[i][j]) > maxVal {
				maxVal = float64(wm.Vec[i][j])
			}
		}

		var sumExp float64
		for j := range wm.Cols {
			sumExp += math.Exp(float64(wm.Vec[i][j]) - maxVal)
		}

		if sumExp == 0 || math.IsInf(sumExp, 0) || math.IsNaN(sumExp) {
			uniform := float32(1.0 / float64(wm.Cols))
			for j := range wm.Cols {
				wm.Vec[i][j] = uniform
			}
			continue
		}

		for j := range wm.Cols {
			wm.Vec[i][j] = float32(math.Exp(float64(wm.Vec[i][j])-maxVal) / sumExp)
		}
	}

	return wm
}

func SoftmaxCopy(wm *Matrix) *Matrix {
	if wm == nil {
		return nil
	}

	out := NewZeroMatrix(wm.Rows, wm.Cols)
	for i := 0; i < wm.Rows; i++ {
		copy(out.Vec[i], wm.Vec[i])
	}

	return Softmax(out)
}

func (wm *Matrix) Transpose() *Matrix {
	n := &Matrix{
		Rows: wm.Cols,
		Cols: wm.Rows,
		Vec:  make([][]float32, wm.Cols),
	}

	for i := 0; i < n.Rows; i++ {
		n.Vec[i] = make([]float32, n.Cols)
		for j := 0; j < n.Cols; j++ {
			n.Vec[i][j] = wm.Vec[j][i]
		}
	}
	return n
}

func (wm *Matrix) Dot(t *Matrix) []float32 {
	if wm.Rows != t.Rows || wm.Cols != t.Cols {
		return nil
	}

	n := make([]float32, wm.Rows)

	for i := range len(wm.Vec) {
		acc := float32(0)
		for j := range len(wm.Vec[i]) {
			acc = acc + wm.Vec[i][j]*t.Vec[i][j]
		}
		n[i] = acc
	}
	return n
}

func (wm *Matrix) Mul(other *Matrix) *Matrix {
	result := New(wm.Rows, other.Cols)

	for i := 0; i < wm.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			var sum float32
			for k := 0; k < wm.Cols; k++ { // shared dimension
				sum += wm.Vec[i][k] * other.Vec[k][j]
			}
			result.Vec[i][j] = sum
		}
	}
	return result
}

func (wm *Matrix) Add(other *Matrix) *Matrix {
	if wm.Cols != other.Cols || wm.Rows != other.Rows {
		return nil
	}

	result := New(wm.Rows, other.Cols)

	for i := 0; i < wm.Rows; i++ {
		for j := 0; j < wm.Cols; j++ {
			result.Vec[i][j] = wm.Vec[i][j] + other.Vec[i][j]
		}
	}
	return result
}

func (wm *Matrix) Div(n float32) *Matrix {
	result := New(wm.Rows, wm.Cols)

	for i := range wm.Rows {
		for j := range wm.Cols {
			result.Vec[i][j] = wm.Vec[i][j] / n
		}
	}
	return result
}

func (wm *Matrix) ReLu() {
	for i := range wm.Rows {
		for j := range wm.Cols {
			if wm.Vec[i][j] < 0 {
				wm.Vec[i][j] = 0
			}
		}
	}
}

func (wm *Matrix) SubScaled(grad *Matrix, scale float32) {
	if wm == nil || grad == nil {
		return
	}
	if wm.Rows != grad.Rows || wm.Cols != grad.Cols {
		return
	}

	for i := 0; i < wm.Rows; i++ {
		for j := 0; j < wm.Cols; j++ {
			wm.Vec[i][j] -= scale * grad.Vec[i][j]
		}
	}
}

func (wm *Matrix) AddInPlace(other *Matrix) {
	if wm == nil || other == nil {
		return
	}
	if wm.Rows != other.Rows || wm.Cols != other.Cols {
		return
	}

	for i := 0; i < wm.Rows; i++ {
		for j := 0; j < wm.Cols; j++ {
			wm.Vec[i][j] += other.Vec[i][j]
		}
	}
}
