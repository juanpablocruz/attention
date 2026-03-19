package transformer

import (
	"math"

	"github.com/juanpablocruz/attention/gen/internal/embbeding"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
)

func AddPosition(pos int, vec *embbeding.EmbeddingVec) *embbeding.EmbeddingVec {
	modelDim := len(vec.Data)
	if modelDim == 0 {
		return &embbeding.EmbeddingVec{}
	}

	positionalVec := make([]float32, modelDim)
	for i := range modelDim {
		pair := i / 2
		val := float64(pos) / math.Pow(10000, float64(2*pair)/float64(modelDim))
		if i&1 == 0 {
			positionalVec[i] = float32(math.Sin(val))
			continue
		}
		positionalVec[i] = float32(math.Cos(val))
	}

	out := embbeding.EmbeddingVec{Data: make([]float32, modelDim)}
	copy(out.Data, vec.Data)
	return out.Add(positionalVec)
}

func ScaledAttention(q *matrix.Matrix, k *matrix.Matrix, v *matrix.Matrix) *matrix.Matrix {
	kT := k.Transpose()
	qkt := q.Mul(kT)
	sqrtD := math.Sqrt(float64(k.Cols))

	f := qkt.Div(float32(sqrtD))

	s := matrix.Softmax(f)
	return s.Mul(v)
}

func ScaledAttentionBackward(dOut, q, k, v *matrix.Matrix) (dQ, dK, dV *matrix.Matrix) {
	if dOut == nil || q == nil || k == nil || v == nil {
		return nil, nil, nil
	}

	sqrtD := float32(math.Sqrt(float64(k.Cols)))

	kT := k.Transpose()
	scores := q.Mul(kT).Div(sqrtD)
	attn := matrix.Softmax(scores)

	dV = attn.Transpose().Mul(dOut)

	dAttn := dOut.Mul(v.Transpose())
	dScores := softmaxBackward(dAttn, attn)

	dQ = dScores.Mul(k).Div(sqrtD)
	dK = dScores.Transpose().Mul(q).Div(sqrtD)

	return dQ, dK, dV
}

func softmaxBackward(dOut, probs *matrix.Matrix) *matrix.Matrix {
	if dOut == nil || probs == nil || dOut.Rows != probs.Rows || dOut.Cols != probs.Cols {
		return nil
	}

	dX := matrix.NewZeroMatrix(dOut.Rows, dOut.Cols)
	for i := 0; i < dOut.Rows; i++ {
		dot := float32(0)
		for j := 0; j < dOut.Cols; j++ {
			dot += dOut.Vec[i][j] * probs.Vec[i][j]
		}

		for j := 0; j < dOut.Cols; j++ {
			dX.Vec[i][j] = probs.Vec[i][j] * (dOut.Vec[i][j] - dot)
		}
	}

	return dX
}

func AddAndNorm(m *matrix.Matrix, attention *matrix.Matrix) *matrix.Matrix {
	return layerNorm(attention.Add(m))
}

func AddAndNormBackward(dOut, m, attention *matrix.Matrix) (dM, dAttention *matrix.Matrix) {
	residual := attention.Add(m)
	dResidual := layerNormBackward(dOut, residual)
	if dResidual == nil {
		return nil, nil
	}

	return cloneMatrix(dResidual), cloneMatrix(dResidual)
}

func layerNorm(residual *matrix.Matrix) *matrix.Matrix {
	means := mean(residual)
	variances := variance(residual, means)
	return normalize(residual, means, variances)
}

func mean(m *matrix.Matrix) []float32 {
	result := make([]float32, m.Rows)
	for i := range m.Rows {
		sum := float32(0.0)
		for j := range m.Cols {
			sum += m.Vec[i][j]
		}
		result[i] = sum / float32(m.Cols)
	}
	return result
}

func variance(m *matrix.Matrix, means []float32) []float32 {
	result := make([]float32, m.Rows)
	for i := range m.Rows {
		sum := float32(0.0)
		for j := range m.Cols {
			sum += float32(math.Pow(float64(m.Vec[i][j]-means[i]), 2))
		}
		result[i] = sum / float32(m.Cols)
	}

	return result
}

func normalize(m *matrix.Matrix, means []float32, variances []float32) *matrix.Matrix {
	result := matrix.New(m.Rows, m.Cols)
	const epsilon float32 = 1e-5
	for i := range m.Rows {
		stdDev := float32(math.Sqrt(float64(variances[i] + epsilon)))
		for j := range m.Cols {
			result.Vec[i][j] = (m.Vec[i][j] - means[i]) / stdDev
		}
	}

	return result
}

type FFNCache struct {
	Input    *matrix.Matrix
	PreRelu  *matrix.Matrix
	Hidden   *matrix.Matrix
	Residual *matrix.Matrix
}

func ExpandWithCache(m, t, w *matrix.Matrix) (*matrix.Matrix, *FFNCache) {
	preRelu := m.Mul(t)

	hidden := matrix.NewZeroMatrix(preRelu.Rows, preRelu.Cols)
	for i := 0; i < preRelu.Rows; i++ {
		copy(hidden.Vec[i], preRelu.Vec[i])
	}
	hidden.ReLu()

	m2 := hidden.Mul(w)
	residual := m2.Add(m)
	out := layerNorm(residual)

	return out, &FFNCache{
		Input:    m,
		PreRelu:  preRelu,
		Hidden:   hidden,
		Residual: residual,
	}
}

func FFNBackward(dOut *matrix.Matrix, cache *FFNCache, w1, w2 *matrix.Matrix) (dW1, dW2, dInput *matrix.Matrix) {
	dResidual := layerNormBackward(dOut, cache.Residual)

	dW2 = cache.Hidden.Transpose().Mul(dResidual)
	dHidden := dResidual.Mul(w2.Transpose())

	dPreRelu := matrix.NewZeroMatrix(dHidden.Rows, dHidden.Cols)
	for i := 0; i < dHidden.Rows; i++ {
		for j := 0; j < dHidden.Cols; j++ {
			if cache.PreRelu.Vec[i][j] > 0 {
				dPreRelu.Vec[i][j] = dHidden.Vec[i][j]
			}
		}
	}

	dW1 = cache.Input.Transpose().Mul(dPreRelu)

	dInputLinear := dPreRelu.Mul(w1.Transpose())
	dInput = dInputLinear.Add(dResidual)

	return dW1, dW2, dInput
}

func layerNormBackward(dOut, x *matrix.Matrix) *matrix.Matrix {
	if dOut == nil || x == nil || dOut.Rows != x.Rows || dOut.Cols != x.Cols {
		return nil
	}

	dX := matrix.NewZeroMatrix(x.Rows, x.Cols)
	const epsilon float32 = 1e-5

	for i := 0; i < x.Rows; i++ {
		n := float32(x.Cols)

		var meanRow float32
		for j := 0; j < x.Cols; j++ {
			meanRow += x.Vec[i][j]
		}
		meanRow /= n

		var varRow float32
		for j := 0; j < x.Cols; j++ {
			d := x.Vec[i][j] - meanRow
			varRow += d * d
		}
		varRow /= n

		invStd := float32(1.0 / math.Sqrt(float64(varRow+epsilon)))

		var sumDOut float32
		var sumDOutXhat float32
		xhat := make([]float32, x.Cols)
		for j := 0; j < x.Cols; j++ {
			xhat[j] = (x.Vec[i][j] - meanRow) * invStd
			sumDOut += dOut.Vec[i][j]
			sumDOutXhat += dOut.Vec[i][j] * xhat[j]
		}

		for j := 0; j < x.Cols; j++ {
			dX.Vec[i][j] = (invStd / n) * (n*dOut.Vec[i][j] - sumDOut - xhat[j]*sumDOutXhat)
		}
	}

	return dX
}

func cloneMatrix(m *matrix.Matrix) *matrix.Matrix {
	if m == nil {
		return nil
	}

	out := matrix.NewZeroMatrix(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		copy(out.Vec[i], m.Vec[i])
	}
	return out
}

func SelectChoice(m *matrix.Matrix) []int {
	result := make([]int, m.Rows)
	for i := 0; i < m.Rows; i++ {
		c := float32(-math.MaxFloat32)
		idx := 0
		for j := 0; j < m.Cols; j++ {
			if math.IsNaN(float64(m.Vec[i][j])) {
				continue
			}
			if m.Vec[i][j] > c {
				c = m.Vec[i][j]
				idx = j
			}
		}
		result[i] = idx
	}
	return result
}
