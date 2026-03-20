package trainer

import (
	"context"
	"fmt"
	"math"

	"github.com/juanpablocruz/attention/gen/internal/attention"
	"github.com/juanpablocruz/attention/gen/internal/intent"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/internal/prompt"
	"github.com/juanpablocruz/attention/gen/internal/transformer"
)

func trainBatch(ctx context.Context, batch []output.Output, model *attention.Model, learningRate float32, gradWorkers int, workCh chan<- batchWork) (float64, int64, int64, int64, int64, int64, int64, int64, int64, error) {
	workerCount := max(min(gradWorkers, len(batch)), 1)
	respCh := make(chan workerBatchAccum, workerCount)
	packed := model.Pack()
	prepared, err := prepareBatch(batch)
	if err != nil {
		return 0, 0, 0, 0, 0, 0, 0, 0, 0, err
	}

	start := 0
	for w := range workerCount {
		remainingWorkers := workerCount - w
		remainingSamples := len(prepared) - start
		chunkSize := (remainingSamples + remainingWorkers - 1) / remainingWorkers
		end := start + chunkSize
		select {
		case <-ctx.Done():
			return 0, 0, 0, 0, 0, 0, 0, 0, 0, context.Canceled
		case workCh <- batchWork{samples: prepared[start:end], packed: packed, resp: respCh}:
		}
		start = end
	}

	agg := newWorkerBatchAccum(model)
	for range workerCount {
		var part workerBatchAccum
		select {
		case <-ctx.Done():
			return 0, 0, 0, 0, 0, 0, 0, 0, 0, context.Canceled
		case part = <-respCh:
		}
		if part.err != nil {
			if part.err == context.Canceled {
				return 0, 0, 0, 0, 0, 0, 0, 0, 0, context.Canceled
			}
			return 0, 0, 0, 0, 0, 0, 0, 0, 0, part.err
		}
		agg.merge(part)
	}

	if agg.count == 0 {
		return 0, 0, 0, 0, 0, 0, 0, 0, 0, fmt.Errorf("empty gradient batch")
	}

	scale := learningRate / float32(agg.count)
	model.Output.SubScaled(agg.dCW, scale)
	model.Block.W2.SubScaled(agg.dW2, scale)
	model.Block.W1.SubScaled(agg.dW1, scale)
	model.Block.WO.SubScaled(agg.dWO, scale)
	for h := 0; h < model.Block.NumHeads; h++ {
		model.Block.WQ[h].SubScaled(agg.dWQHeads[h], scale)
		model.Block.WK[h].SubScaled(agg.dWKHeads[h], scale)
		model.Block.WV[h].SubScaled(agg.dWVHeads[h], scale)
	}
	model.Embedding.SubScaledByGrad(agg.dEmbedding, scale)

	return agg.loss, agg.correct, agg.tokens, agg.sortCorrect, agg.sortTokens, agg.sumCorrect, agg.sumTokens, agg.sortSamples, agg.sumSamples, nil
}

func prepareBatch(batch []output.Output) ([]preparedSample, error) {
	prepared := make([]preparedSample, len(batch))
	for i, out := range batch {
		numbers, err := prompt.ExtractNumbers(out.Prompt)
		if err != nil {
			return nil, err
		}
		label, err := labelFromRecord(out.Intent)
		if err != nil {
			return nil, err
		}
		task, order, err := intent.LabelToTaskOrder(label)
		if err != nil {
			return nil, err
		}
		sourceTokens, err := prompt.EncodeStructured(task, order, numbers)
		if err != nil {
			return nil, err
		}
		targetNumbers, err := prompt.ExtractNumbers(out.Target)
		if err != nil {
			return nil, err
		}
		targetTokens, err := prompt.EncodeStructured(task, order, targetNumbers)
		if err != nil {
			return nil, err
		}
		prepared[i] = preparedSample{
			source: sourceTokens,
			target: targetTokens,
			task:   task,
			count:  len(numbers),
		}
	}
	return prepared, nil
}

func newWorkerBatchAccum(model *attention.Model) workerBatchAccum {
	acc := workerBatchAccum{
		dCW:        matrix.NewZeroMatrix(model.Output.Rows, model.Output.Cols),
		dW1:        matrix.NewZeroMatrix(model.Block.W1.Rows, model.Block.W1.Cols),
		dW2:        matrix.NewZeroMatrix(model.Block.W2.Rows, model.Block.W2.Cols),
		dWO:        matrix.NewZeroMatrix(model.Block.WO.Rows, model.Block.WO.Cols),
		dWQHeads:   make([]*matrix.Matrix, model.Block.NumHeads),
		dWKHeads:   make([]*matrix.Matrix, model.Block.NumHeads),
		dWVHeads:   make([]*matrix.Matrix, model.Block.NumHeads),
		dEmbedding: matrix.NewZeroMatrix(model.Embedding.Vocab, model.Embedding.Dimensions),
	}
	for h := 0; h < model.Block.NumHeads; h++ {
		acc.dWQHeads[h] = matrix.NewZeroMatrix(model.Block.WQ[h].Rows, model.Block.WQ[h].Cols)
		acc.dWKHeads[h] = matrix.NewZeroMatrix(model.Block.WK[h].Rows, model.Block.WK[h].Cols)
		acc.dWVHeads[h] = matrix.NewZeroMatrix(model.Block.WV[h].Rows, model.Block.WV[h].Cols)
	}
	return acc
}

func (a *workerBatchAccum) add(grad *sampleGrad) {
	a.loss += float64(grad.loss)
	a.correct += int64(grad.correct)
	a.tokens += int64(grad.tokens)
	a.sortCorrect += int64(grad.taskCorrectIfSort())
	a.sortTokens += int64(grad.taskTokensIfSort())
	a.sumCorrect += int64(grad.taskCorrectIfSum())
	a.sumTokens += int64(grad.taskTokensIfSum())
	a.sortSamples += int64(grad.sampleIfSort())
	a.sumSamples += int64(grad.sampleIfSum())
	a.count++

	a.dCW.AddInPlace(grad.dCW)
	a.dW1.AddInPlace(grad.dW1)
	a.dW2.AddInPlace(grad.dW2)
	a.dWO.AddInPlace(grad.dWO)
	for h := 0; h < len(a.dWQHeads); h++ {
		a.dWQHeads[h].AddInPlace(grad.dWQHeads[h])
		a.dWKHeads[h].AddInPlace(grad.dWKHeads[h])
		a.dWVHeads[h].AddInPlace(grad.dWVHeads[h])
	}
	a.dEmbedding.AddInPlace(grad.dEmbedding)
}

func (a *workerBatchAccum) merge(other workerBatchAccum) {
	a.loss += other.loss
	a.correct += other.correct
	a.tokens += other.tokens
	a.sortCorrect += other.sortCorrect
	a.sortTokens += other.sortTokens
	a.sumCorrect += other.sumCorrect
	a.sumTokens += other.sumTokens
	a.sortSamples += other.sortSamples
	a.sumSamples += other.sumSamples
	a.count += other.count

	a.dCW.AddInPlace(other.dCW)
	a.dW1.AddInPlace(other.dW1)
	a.dW2.AddInPlace(other.dW2)
	a.dWO.AddInPlace(other.dWO)
	for h := 0; h < len(a.dWQHeads); h++ {
		a.dWQHeads[h].AddInPlace(other.dWQHeads[h])
		a.dWKHeads[h].AddInPlace(other.dWKHeads[h])
		a.dWVHeads[h].AddInPlace(other.dWVHeads[h])
	}
	a.dEmbedding.AddInPlace(other.dEmbedding)
}

func computeSampleGrad(source []uint8, target []uint8, task string, count int, model *attention.Model, packed *attention.PackedModel) (*sampleGrad, error) {
	pass, err := model.ForwardPacked(source, packed)
	if err != nil {
		return nil, err
	}
	correct, measuredTokens := scorePrediction(pass.Choices, target, task, count)

	lossValue, dLogits, err := taskCrossEntropyWithGrad(pass.Logits, target, task, count)
	if err != nil {
		return nil, err
	}

	var packedBlock *transformer.PackedBlock
	var dFFN *matrix.Matrix
	if packed != nil {
		packedBlock = packed.Block
	}
	if packed != nil && packed.OutputTranspose != nil {
		dFFN = dLogits.MulPackedInto(nil, packed.OutputTranspose)
	} else {
		dFFN = dLogits.Mul(model.Output.Transpose())
	}
	dCW := pass.FFN.Transpose().Mul(dLogits)

	var packedW1T, packedW2T *matrix.PackedMatrix
	if packedBlock != nil {
		packedW1T = packedBlock.W1Transpose
		packedW2T = packedBlock.W2Transpose
	}
	dW1, dW2, dOutFromFFN := transformer.FFNBackwardPacked(dFFN, pass.FFNCache, model.Block.W1, model.Block.W2, packedW1T, packedW2T)
	dSequenceSkip, dAttention := transformer.AddAndNormBackward(dOutFromFFN, pass.Sequence, pass.Attention)

	dWQHeads, dWKHeads, dWVHeads, dWO, dSequenceFromAttention, err := transformer.MultiHeadAttentionBackwardPacked(dAttention, pass.MHACache, model.Block, packedBlock)
	if err != nil {
		return nil, err
	}
	dSequence := dSequenceSkip.Add(dSequenceFromAttention)

	dEmbedding := matrix.NewZeroMatrix(model.Embedding.Vocab, model.Embedding.Dimensions)
	for pos := range len(source) {
		token := int(source[pos])
		if token < 0 || token >= model.Embedding.Vocab {
			return nil, fmt.Errorf("token out of vocab: %d", token)
		}
		for j := 0; j < model.Embedding.Dimensions; j++ {
			dEmbedding.Vec[token][j] += dSequence.Vec[pos][j]
		}
	}

	return &sampleGrad{
		loss:        lossValue,
		correct:     correct,
		tokens:      measuredTokens,
		task:        task,
		taskCorrect: correct,
		taskTokens:  measuredTokens,
		dCW:         dCW,
		dW1:         dW1,
		dW2:         dW2,
		dWO:         dWO,
		dWQHeads:    dWQHeads,
		dWKHeads:    dWKHeads,
		dWVHeads:    dWVHeads,
		dEmbedding:  dEmbedding,
	}, nil
}

func taskCrossEntropyWithGrad(logits *matrix.Matrix, target []uint8, task string, count int) (float32, *matrix.Matrix, error) {
	if logits == nil || logits.Rows == 0 || logits.Cols == 0 {
		return 0, matrix.NewZeroMatrix(0, 0), nil
	}
	if len(target) != logits.Rows {
		return 0, nil, fmt.Errorf("target length %d must match logits rows %d", len(target), logits.Rows)
	}

	indices := make([]int, 0, 4)
	indices = append(indices, 0, 1)
	if task == prompt.TaskSum {
		indices = append(indices, 2+prompt.MaxListLen-2, 2+prompt.MaxListLen-1)
	} else {
		for i := range count {
			indices = append(indices, 2+i)
		}
		indices = append(indices, prompt.SequenceLen-1)
	}

	probs := matrix.SoftmaxCopy(logits)
	grad := matrix.NewZeroMatrix(logits.Rows, logits.Cols)

	var totalLoss float64
	const eps = 1e-9
	for _, i := range indices {
		if i < 0 || i >= logits.Rows {
			continue
		}
		t := int(target[i])
		if t < 0 || t >= logits.Cols {
			return 0, nil, fmt.Errorf("target token %d out of range at row %d", t, i)
		}

		for j := 0; j < logits.Cols; j++ {
			grad.Vec[i][j] = probs.Vec[i][j]
		}
		p := probs.Vec[i][t]
		if p < eps {
			p = eps
		}
		totalLoss += -math.Log(float64(p))
		grad.Vec[i][t] -= 1
	}

	if len(indices) == 0 {
		return 0, grad, nil
	}

	scale := float32(1.0 / float64(len(indices)))
	if task == prompt.TaskSum {
		scale *= 2.5
	}
	for r := 0; r < grad.Rows; r++ {
		for c := 0; c < grad.Cols; c++ {
			grad.Vec[r][c] *= scale
		}
	}
	lossScale := float32(1.0 / float64(len(indices)))
	if task == prompt.TaskSum {
		lossScale *= 2.5
	}
	return float32(totalLoss) * lossScale, grad, nil
}

func (g *sampleGrad) taskCorrectIfSort() int {
	if g.task == prompt.TaskSort {
		return g.taskCorrect
	}
	return 0
}

func (g *sampleGrad) taskTokensIfSort() int {
	if g.task == prompt.TaskSort {
		return g.taskTokens
	}
	return 0
}

func (g *sampleGrad) taskCorrectIfSum() int {
	if g.task == prompt.TaskSum {
		return g.taskCorrect
	}
	return 0
}

func (g *sampleGrad) taskTokensIfSum() int {
	if g.task == prompt.TaskSum {
		return g.taskTokens
	}
	return 0
}

func (g *sampleGrad) sampleIfSort() int {
	if g.task == prompt.TaskSort {
		return 1
	}
	return 0
}

func (g *sampleGrad) sampleIfSum() int {
	if g.task == prompt.TaskSum {
		return 1
	}
	return 0
}

func scorePrediction(pred []int, target []uint8, task string, count int) (int, int) {
	correct := 0
	tokens := 0

	if task == prompt.TaskSum {
		for _, idx := range []int{2 + prompt.MaxListLen - 2, 2 + prompt.MaxListLen - 1} {
			tokens++
			if uint8(pred[idx]) == target[idx] {
				correct++
			}
		}
		return correct, tokens
	}

	for i := range count {
		idx := 2 + i
		tokens++
		if uint8(pred[idx]) == target[idx] {
			correct++
		}
	}

	return correct, tokens
}
