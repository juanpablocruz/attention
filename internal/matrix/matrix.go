package matrix

import (
	"math"
	"math/rand/v2"
)

type Matrix struct {
	Data []float32
	Vec  [][]float32
	Cols int
	Rows int
}

func New(rows, cols int) *Matrix {
	data := make([]float32, rows*cols)
	vec := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		row := data[i*cols : (i+1)*cols]
		for j := 0; j < cols; j++ {
			row[j] = rand.Float32()*0.2 - 0.1
		}
		vec[i] = row
	}
	return &Matrix{Data: data, Vec: vec, Rows: rows, Cols: cols}
}

func NewZeroMatrix(rows, cols int) *Matrix {
	data := make([]float32, rows*cols)
	vec := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		vec[i] = data[i*cols : (i+1)*cols]
	}
	return &Matrix{Data: data, Vec: vec, Rows: rows, Cols: cols}
}

func Softmax(wm *Matrix) *Matrix {
	if wm == nil || wm.Rows == 0 || wm.Cols == 0 {
		return wm
	}

	for i := range wm.Rows {
		row := wm.Vec[i]
		maxVal := float64(simdRowMax(row))

		var sumExp float64
		sumExp = float64(simdRowSumExpShift(row, float32(maxVal)))

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
	n := NewZeroMatrix(wm.Cols, wm.Rows)

	for i := 0; i < n.Rows; i++ {
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
	if wm == nil || other == nil {
		return nil
	}
	if wm.Cols != other.Rows {
		return nil
	}
	return mulNeon(wm, other)
}

func (wm *Matrix) Add(other *Matrix) *Matrix {
	if wm.Cols != other.Cols || wm.Rows != other.Rows {
		return nil
	}

	result := NewZeroMatrix(wm.Rows, other.Cols)

	for i := 0; i < wm.Rows; i++ {
		for j := 0; j < wm.Cols; j++ {
			result.Vec[i][j] = wm.Vec[i][j] + other.Vec[i][j]
		}
	}
	return result
}

func (wm *Matrix) Div(n float32) *Matrix {
	result := NewZeroMatrix(wm.Rows, wm.Cols)

	for i := range wm.Rows {
		for j := range wm.Cols {
			result.Vec[i][j] = wm.Vec[i][j] / n
		}
	}
	return result
}

func (wm *Matrix) ReLu() {
	if wm == nil || len(wm.Data) == 0 {
		return
	}
	simdReLU(wm.Data)
}

func (wm *Matrix) SubScaled(grad *Matrix, scale float32) {
	if wm == nil || grad == nil {
		return
	}
	if wm.Rows != grad.Rows || wm.Cols != grad.Cols {
		return
	}
	if len(wm.Data) == wm.Rows*wm.Cols && len(grad.Data) == grad.Rows*grad.Cols {
		simdSubScaled(wm.Data, grad.Data, scale)
		return
	}
	for i := 0; i < wm.Rows; i++ {
		simdSubScaled(wm.Vec[i], grad.Vec[i], scale)
	}
}

func (wm *Matrix) AddInPlace(other *Matrix) {
	if wm == nil || other == nil {
		return
	}
	if wm.Rows != other.Rows || wm.Cols != other.Cols {
		return
	}
	if len(wm.Data) == wm.Rows*wm.Cols && len(other.Data) == other.Rows*other.Cols {
		simdAddInPlace(wm.Data, other.Data)
		return
	}
	for i := 0; i < wm.Rows; i++ {
		simdAddInPlace(wm.Vec[i], other.Vec[i])
	}
}

func RowMean(row []float32) float32 {
	return simdRowMean(row)
}

func RowVariance(row []float32, mean float32) float32 {
	return simdRowVariance(row, mean)
}

func NormalizeRow(dst, src []float32, mean, invStd float32) {
	simdNormalizeRow(dst, src, mean, invStd)
}
