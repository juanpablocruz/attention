package transformer

import (
	"fmt"
	"math"

	"github.com/juanpablocruz/attention/gen/internal/embbeding"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
)

type HeadCache struct {
	Q, K, V *matrix.Matrix
}

type MHACache struct {
	Input  *matrix.Matrix
	Concat *matrix.Matrix
	Heads  []HeadCache
}

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

func MultiHeadAttention(x *matrix.Matrix, block *TransformerBlock) (*matrix.Matrix, *MHACache, error) {
	return MultiHeadAttentionPacked(x, block, nil)
}

func MultiHeadAttentionPacked(x *matrix.Matrix, block *TransformerBlock, packed *PackedBlock) (*matrix.Matrix, *MHACache, error) {
	if x == nil || block == nil {
		return nil, nil, fmt.Errorf("nil input for multi-head attention")
	}
	if len(block.WQ) != block.NumHeads || len(block.WK) != block.NumHeads || len(block.WV) != block.NumHeads {
		return nil, nil, fmt.Errorf("inconsistent number of attention heads")
	}

	headOuts := make([]*matrix.Matrix, block.NumHeads)
	headCaches := make([]HeadCache, block.NumHeads)
	for h := 0; h < block.NumHeads; h++ {
		var q, k, v *matrix.Matrix
		if packed != nil {
			q = x.MulPackedInto(nil, packed.WQ[h])
			k = x.MulPackedInto(nil, packed.WK[h])
			v = x.MulPackedInto(nil, packed.WV[h])
		} else {
			q = x.Mul(block.WQ[h])
			k = x.Mul(block.WK[h])
			v = x.Mul(block.WV[h])
		}

		headOuts[h] = ScaledAttention(q, k, v)
		headCaches[h] = HeadCache{Q: q, K: k, V: v}
	}

	concat := concatHeads(headOuts)
	var out *matrix.Matrix
	if packed != nil {
		out = concat.MulPackedInto(nil, packed.WO)
	} else {
		out = concat.Mul(block.WO)
	}

	return out, &MHACache{Input: x, Concat: concat, Heads: headCaches}, nil
}

func MultiHeadAttentionBackward(dOut *matrix.Matrix, cache *MHACache, block *TransformerBlock) (dWQ, dWK, dWV []*matrix.Matrix, dWO, dInput *matrix.Matrix, err error) {
	return MultiHeadAttentionBackwardPacked(dOut, cache, block, nil)
}

func MultiHeadAttentionBackwardPacked(dOut *matrix.Matrix, cache *MHACache, block *TransformerBlock, packed *PackedBlock) (dWQ, dWK, dWV []*matrix.Matrix, dWO, dInput *matrix.Matrix, err error) {
	if dOut == nil || cache == nil || block == nil {
		return nil, nil, nil, nil, nil, fmt.Errorf("nil inputs for multi-head backward")
	}

	concatT := cache.Concat.Transpose()
	inputT := cache.Input.Transpose()

	dWO = concatT.Mul(dOut)
	var dConcat *matrix.Matrix
	if packed != nil {
		dConcat = dOut.MulPackedInto(nil, packed.WOTranspose)
	} else {
		dConcat = dOut.Mul(block.WO.Transpose())
	}
	dHeadOuts, splitErr := splitHeads(dConcat, block.NumHeads, block.HeadDim)
	if splitErr != nil {
		return nil, nil, nil, nil, nil, splitErr
	}

	dWQ = make([]*matrix.Matrix, block.NumHeads)
	dWK = make([]*matrix.Matrix, block.NumHeads)
	dWV = make([]*matrix.Matrix, block.NumHeads)
	dInput = matrix.NewZeroMatrix(cache.Input.Rows, cache.Input.Cols)

	for h := 0; h < block.NumHeads; h++ {
		dQ, dK, dV := ScaledAttentionBackward(dHeadOuts[h], cache.Heads[h].Q, cache.Heads[h].K, cache.Heads[h].V)

		dWQ[h] = inputT.Mul(dQ)
		dWK[h] = inputT.Mul(dK)
		dWV[h] = inputT.Mul(dV)

		var dXQ, dXK, dXV *matrix.Matrix
		if packed != nil {
			dXQ = dQ.MulPackedInto(nil, packed.WQTranspose[h])
			dXK = dK.MulPackedInto(nil, packed.WKTranspose[h])
			dXV = dV.MulPackedInto(nil, packed.WVTranspose[h])
		} else {
			dXQ = dQ.Mul(block.WQ[h].Transpose())
			dXK = dK.Mul(block.WK[h].Transpose())
			dXV = dV.Mul(block.WV[h].Transpose())
		}

		dInput.AddInPlace(dXQ)
		dInput.AddInPlace(dXK)
		dInput.AddInPlace(dXV)
	}

	return dWQ, dWK, dWV, dWO, dInput, nil
}

func concatHeads(headOuts []*matrix.Matrix) *matrix.Matrix {
	rows := headOuts[0].Rows
	totalCols := 0
	for _, h := range headOuts {
		totalCols += h.Cols
	}

	out := matrix.NewZeroMatrix(rows, totalCols)
	colOffset := 0
	for _, h := range headOuts {
		for i := range rows {
			copy(out.Vec[i][colOffset:colOffset+h.Cols], h.Vec[i])
		}
		colOffset += h.Cols
	}

	return out
}

func splitHeads(concat *matrix.Matrix, numHeads, headDim int) ([]*matrix.Matrix, error) {
	if concat.Cols != numHeads*headDim {
		return nil, fmt.Errorf("invalid concat shape for head split: got %d cols, expected %d", concat.Cols, numHeads*headDim)
	}

	out := make([]*matrix.Matrix, numHeads)
	for h := range numHeads {
		part := matrix.NewZeroMatrix(concat.Rows, headDim)
		start := h * headDim
		end := start + headDim
		for i := 0; i < concat.Rows; i++ {
			copy(part.Vec[i], concat.Vec[i][start:end])
		}
		out[h] = part
	}

	return out, nil
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

	return dResidual, dResidual
}

func layerNorm(residual *matrix.Matrix) *matrix.Matrix {
	means := mean(residual)
	variances := variance(residual, means)
	return normalize(residual, means, variances)
}

func mean(m *matrix.Matrix) []float32 {
	result := make([]float32, m.Rows)
	for i := range m.Rows {
		result[i] = matrix.RowMean(m.Vec[i])
	}
	return result
}

func variance(m *matrix.Matrix, means []float32) []float32 {
	result := make([]float32, m.Rows)
	for i := range m.Rows {
		result[i] = matrix.RowVariance(m.Vec[i], means[i])
	}

	return result
}

func normalize(m *matrix.Matrix, means []float32, variances []float32) *matrix.Matrix {
	result := matrix.NewZeroMatrix(m.Rows, m.Cols)
	const epsilon float32 = 1e-5
	for i := range m.Rows {
		stdDev := float32(math.Sqrt(float64(variances[i] + epsilon)))
		matrix.NormalizeRow(result.Vec[i], m.Vec[i], means[i], 1/stdDev)
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
	return ExpandWithCachePacked(m, t, w, nil, nil)
}

func ExpandWithCachePacked(m, t, w *matrix.Matrix, packedT, packedW *matrix.PackedMatrix) (*matrix.Matrix, *FFNCache) {
	var preRelu *matrix.Matrix
	if packedT != nil {
		preRelu = m.MulPackedInto(nil, packedT)
	} else {
		preRelu = m.Mul(t)
	}

	hidden := matrix.NewZeroMatrix(preRelu.Rows, preRelu.Cols)
	copy(hidden.Data, preRelu.Data)
	hidden.ReLu()

	var m2 *matrix.Matrix
	if packedW != nil {
		m2 = hidden.MulPackedInto(nil, packedW)
	} else {
		m2 = hidden.Mul(w)
	}
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
	return FFNBackwardPacked(dOut, cache, w1, w2, nil, nil)
}

func FFNBackwardPacked(dOut *matrix.Matrix, cache *FFNCache, w1, w2 *matrix.Matrix, packedW1T, packedW2T *matrix.PackedMatrix) (dW1, dW2, dInput *matrix.Matrix) {
	dResidual := layerNormBackward(dOut, cache.Residual)
	hiddenT := cache.Hidden.Transpose()
	inputT := cache.Input.Transpose()

	dW2 = hiddenT.Mul(dResidual)
	var dHidden *matrix.Matrix
	if packedW2T != nil {
		dHidden = dResidual.MulPackedInto(nil, packedW2T)
	} else {
		dHidden = dResidual.Mul(w2.Transpose())
	}

	dPreRelu := matrix.NewZeroMatrix(dHidden.Rows, dHidden.Cols)
	for i := 0; i < dHidden.Rows; i++ {
		for j := 0; j < dHidden.Cols; j++ {
			if cache.PreRelu.Vec[i][j] > 0 {
				dPreRelu.Vec[i][j] = dHidden.Vec[i][j]
			}
		}
	}

	dW1 = inputT.Mul(dPreRelu)

	var dInputLinear *matrix.Matrix
	if packedW1T != nil {
		dInputLinear = dPreRelu.MulPackedInto(nil, packedW1T)
	} else {
		dInputLinear = dPreRelu.Mul(w1.Transpose())
	}
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
	return m.CloneInto(nil)
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
