package trainer

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/juanpablocruz/attention/gen/internal/checkpoint"
	"github.com/juanpablocruz/attention/gen/internal/embbeding"
	lossfn "github.com/juanpablocruz/attention/gen/internal/loss"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/internal/prompt"
	"github.com/juanpablocruz/attention/gen/internal/transformer"
	"github.com/juanpablocruz/attention/gen/pkg/decode"
)

const (
	defaultEpochs       = 3
	defaultCheckpoint   = "./checkpoints/embed_model.bin"
	defaultVocabSize    = prompt.VocabSize
	defaultModelDim     = 64
	defaultNumHeads     = 4
	defaultLearningRate = float32(0.01)
	defaultBatchSize    = 256
	progressInterval    = 2 * time.Second
	envBatchSize        = "ATTN_BATCH_SIZE"
	envGradWorkers      = "ATTN_GRAD_WORKERS"
	envNumHeads         = "ATTN_NUM_HEADS"
	envModelDim         = "ATTN_MODEL_DIM"
)

type Config struct {
	DatasetPath    string
	Epochs         int
	CheckpointPath string
	VocabSize      int
	ModelDim       int
	NumHeads       int
	LearningRate   float32
	BatchSize      int
	GradWorkers    int
}

func (c Config) CheckpointPathOrDefault() string {
	if c.CheckpointPath != "" {
		return c.CheckpointPath
	}
	return defaultCheckpoint
}

type EpochMetrics struct {
	Epoch        int
	TotalEpochs  int
	Samples      int64
	Loss         float64
	TokenAcc     float64
	SortTokenAcc float64
	SumTokenAcc  float64
	SortSamples  int64
	SumSamples   int64
	Elapsed      time.Duration
}

type outputSink struct {
	ctx context.Context
	ch  chan<- output.Output
}

type sampleGrad struct {
	loss        float32
	correct     int
	tokens      int
	task        string
	taskCorrect int
	taskTokens  int
	dCW         *matrix.Matrix
	dW1         *matrix.Matrix
	dW2         *matrix.Matrix
	dWO         *matrix.Matrix
	dWQHeads    []*matrix.Matrix
	dWKHeads    []*matrix.Matrix
	dWVHeads    []*matrix.Matrix
	dEmbedding  *matrix.Matrix
}

type batchWork struct {
	samples []output.Output
	resp    chan<- workerBatchAccum
}

type workerBatchAccum struct {
	err         error
	loss        float64
	correct     int64
	tokens      int64
	sortCorrect int64
	sortTokens  int64
	sumCorrect  int64
	sumTokens   int64
	sortSamples int64
	sumSamples  int64
	count       int
	dCW         *matrix.Matrix
	dW1         *matrix.Matrix
	dW2         *matrix.Matrix
	dWO         *matrix.Matrix
	dWQHeads    []*matrix.Matrix
	dWKHeads    []*matrix.Matrix
	dWVHeads    []*matrix.Matrix
	dEmbedding  *matrix.Matrix
}

func (s *outputSink) Process(o output.Output) {
	select {
	case <-s.ctx.Done():
		return
	case s.ch <- o:
	}
}

func Train(ctx context.Context, cfg Config) (bool, []EpochMetrics, bool, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	cfg = normalizeConfig(cfg)

	m := embbeding.New(cfg.VocabSize, cfg.ModelDim)
	tBlock := transformer.New(cfg.ModelDim, cfg.NumHeads)
	cW := matrix.New(cfg.ModelDim, cfg.VocabSize)

	loaded, err := checkpoint.Load(cfg.CheckpointPath, m, tBlock, cW)
	if err != nil {
		loaded = false
	}

	metrics := make([]EpochMetrics, 0, cfg.Epochs)
	interrupted := false
	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if ctx.Err() != nil {
			interrupted = true
			break
		}

		start := time.Now()
		avgLoss, tokenAcc, seen, sortAcc, sumAcc, sortSamples, sumSamples, canceled, err := trainEpoch(ctx, epoch, cfg.Epochs, cfg.DatasetPath, m, tBlock, cW, cfg.LearningRate, cfg.ModelDim, cfg.BatchSize, cfg.GradWorkers)
		if err != nil {
			return loaded, nil, interrupted, err
		}
		if canceled {
			interrupted = true
		}

		if err := checkpoint.Save(cfg.CheckpointPath, m, tBlock, cW); err != nil {
			return loaded, nil, interrupted, err
		}

		metrics = append(metrics, EpochMetrics{
			Epoch:        epoch,
			TotalEpochs:  cfg.Epochs,
			Samples:      seen,
			Loss:         avgLoss,
			TokenAcc:     tokenAcc,
			SortTokenAcc: sortAcc,
			SumTokenAcc:  sumAcc,
			SortSamples:  sortSamples,
			SumSamples:   sumSamples,
			Elapsed:      time.Since(start),
		})

		if canceled {
			break
		}
	}

	return loaded, metrics, interrupted, nil
}

func normalizeConfig(cfg Config) Config {
	if cfg.Epochs <= 0 {
		cfg.Epochs = defaultEpochs
	}
	if cfg.CheckpointPath == "" {
		cfg.CheckpointPath = defaultCheckpoint
	}
	if cfg.VocabSize <= 0 {
		cfg.VocabSize = defaultVocabSize
	}
	if cfg.ModelDim <= 0 {
		cfg.ModelDim = defaultModelDim
	}
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = defaultLearningRate
	}
	if cfg.NumHeads <= 0 {
		cfg.NumHeads = defaultNumHeads
	}
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = defaultBatchSize
	}
	if cfg.GradWorkers <= 0 {
		cfg.GradWorkers = runtime.NumCPU()
	}
	if cfg.GradWorkers < 1 {
		cfg.GradWorkers = 1
	}
	if v := os.Getenv(envBatchSize); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			cfg.BatchSize = parsed
		}
	}
	if v := os.Getenv(envGradWorkers); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			cfg.GradWorkers = parsed
		}
	}
	if v := os.Getenv(envModelDim); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			cfg.ModelDim = parsed
		}
	}
	if v := os.Getenv(envNumHeads); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			cfg.NumHeads = parsed
		}
	}

	return cfg
}

func trainEpoch(ctx context.Context, epoch, totalEpochs int, datasetPath string, m *embbeding.Embedding, tBlock *transformer.TransformerBlock, cW *matrix.Matrix, learningRate float32, modelDim, batchSize, gradWorkers int) (float64, float64, int64, float64, float64, int64, int64, bool, error) {
	f, totalRecords, workers, err := openDataset(datasetPath)
	if err != nil {
		return 0, 0, 0, 0, 0, 0, 0, false, err
	}
	defer f.Close()

	outs := make(chan output.Output, 4096)
	proc := &outputSink{ctx: ctx, ch: outs}

	var wg sync.WaitGroup
	wg.Add(workers)
	decode.DecodeOutput(ctx, int64(workers), totalRecords, &wg, f, proc)

	go func() {
		wg.Wait()
		close(outs)
	}()

	var totalLoss float64
	var totalCorrect int64
	var totalTokens int64
	var totalSortCorrect int64
	var totalSortTokens int64
	var totalSumCorrect int64
	var totalSumTokens int64
	var totalSortSamples int64
	var totalSumSamples int64
	var seen int64
	batch := make([]output.Output, 0, batchSize)

	workCh := make(chan batchWork, gradWorkers)
	var gradWG sync.WaitGroup
	gradWG.Add(gradWorkers)
	for range gradWorkers {
		go func() {
			defer gradWG.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case work, ok := <-workCh:
					if !ok {
						return
					}

					accum := newWorkerBatchAccum(m, tBlock, cW)
					for _, out := range work.samples {
						if ctx.Err() != nil {
							accum.err = ctx.Err()
							break
						}

						parsedSource, err := prompt.Parse(out.Prompt)
						if err != nil {
							accum.err = err
							break
						}
						sourceTokens, err := prompt.DefaultTokenizer.EncodeParsed(parsedSource)
						if err != nil {
							accum.err = err
							break
						}
						targetTokens, err := prompt.Encode(out.Target)
						if err != nil {
							accum.err = err
							break
						}
						grad, err := computeSampleGrad(sourceTokens[:], targetTokens[:], parsedSource, m, tBlock, cW, modelDim)
						if err != nil {
							accum.err = err
							break
						}
						accum.add(grad)
					}
					work.resp <- accum
				}
			}
		}()
	}
	defer func() {
		close(workCh)
		gradWG.Wait()
	}()
	epochStart := time.Now()
	lastReport := epochStart
	spinner := []rune{'|', '/', '-', '\\'}
	spinnerIdx := 0
	didReport := false

	canceled := false
	for {
		if ctx.Err() != nil {
			canceled = true
			break
		}

		out, ok := <-outs
		if !ok {
			break
		}

		batch = append(batch, out)
		if len(batch) < batchSize {
			continue
		}

		batchLoss, batchCorrect, batchTokens, aggSortCorrect, aggSortTokens, aggSumCorrect, aggSumTokens, aggSortSamples, aggSumSamples, err := trainBatch(ctx, batch, m, tBlock, cW, learningRate, modelDim, gradWorkers, workCh)
		if err != nil {
			if err == context.Canceled {
				canceled = true
				break
			}
			return 0, 0, 0, 0, 0, 0, 0, canceled, err
		}

		totalLoss += batchLoss
		totalCorrect += batchCorrect
		totalTokens += batchTokens
		totalSortCorrect += aggSortCorrect
		totalSortTokens += aggSortTokens
		totalSumCorrect += aggSumCorrect
		totalSumTokens += aggSumTokens
		totalSortSamples += aggSortSamples
		totalSumSamples += aggSumSamples
		seen += int64(len(batch))
		batch = batch[:0]

		now := time.Now()
		if now.Sub(lastReport) >= progressInterval {
			avgLoss := totalLoss / float64(max(seen, 1))
			tokenAcc := float64(totalCorrect) / float64(max(totalTokens, 1))
			percent := 100.0 * float64(seen) / float64(totalRecords)
			elapsed := now.Sub(epochStart)
			rate := float64(seen) / elapsed.Seconds()
			remaining := float64(totalRecords-seen) / maxFloat(rate, 1e-9)
			eta := time.Duration(remaining * float64(time.Second))

			fmt.Printf("\r%c epoch=%d/%d progress=%.2f%% samples=%d/%d loss=%.6f token_acc=%.4f speed=%.0f/s eta=%s", spinner[spinnerIdx], epoch, totalEpochs, percent, seen, totalRecords, avgLoss, tokenAcc, rate, eta.Round(time.Second))
			spinnerIdx = (spinnerIdx + 1) % len(spinner)
			lastReport = now
			didReport = true
		}
	}

	if len(batch) > 0 {
		batchLoss, batchCorrect, batchTokens, aggSortCorrect, aggSortTokens, aggSumCorrect, aggSumTokens, aggSortSamples, aggSumSamples, err := trainBatch(ctx, batch, m, tBlock, cW, learningRate, modelDim, gradWorkers, workCh)
		if err != nil {
			if err == context.Canceled {
				canceled = true
			} else {
				return 0, 0, 0, 0, 0, 0, 0, canceled, err
			}
		} else {
			totalLoss += batchLoss
			totalCorrect += batchCorrect
			totalTokens += batchTokens
			totalSortCorrect += aggSortCorrect
			totalSortTokens += aggSortTokens
			totalSumCorrect += aggSumCorrect
			totalSumTokens += aggSumTokens
			totalSortSamples += aggSortSamples
			totalSumSamples += aggSumSamples
			seen += int64(len(batch))
		}
	}

	if didReport {
		fmt.Print("\n")
	}

	if seen == 0 {
		if canceled {
			return 0, 0, 0, 0, 0, 0, 0, true, nil
		}
		return 0, 0, 0, 0, 0, 0, 0, canceled, fmt.Errorf("dataset has no records")
	}

	sortAcc := 0.0
	if totalSortTokens > 0 {
		sortAcc = float64(totalSortCorrect) / float64(totalSortTokens)
	}
	sumAcc := 0.0
	if totalSumTokens > 0 {
		sumAcc = float64(totalSumCorrect) / float64(totalSumTokens)
	}

	return totalLoss / float64(seen), float64(totalCorrect) / float64(totalTokens), seen, sortAcc, sumAcc, totalSortSamples, totalSumSamples, canceled, nil
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func trainBatch(ctx context.Context, batch []output.Output, m *embbeding.Embedding, tBlock *transformer.TransformerBlock, cW *matrix.Matrix, learningRate float32, modelDim, gradWorkers int, workCh chan<- batchWork) (float64, int64, int64, int64, int64, int64, int64, int64, int64, error) {
	workerCount := max(min(gradWorkers, len(batch)), 1)
	respCh := make(chan workerBatchAccum, workerCount)

	start := 0
	for w := 0; w < workerCount; w++ {
		remainingWorkers := workerCount - w
		remainingSamples := len(batch) - start
		chunkSize := (remainingSamples + remainingWorkers - 1) / remainingWorkers
		end := start + chunkSize
		select {
		case <-ctx.Done():
			return 0, 0, 0, 0, 0, 0, 0, 0, 0, context.Canceled
		case workCh <- batchWork{samples: batch[start:end], resp: respCh}:
		}
		start = end
	}

	agg := newWorkerBatchAccum(m, tBlock, cW)
	for i := 0; i < workerCount; i++ {
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
	cW.SubScaled(agg.dCW, scale)
	tBlock.W2.SubScaled(agg.dW2, scale)
	tBlock.W1.SubScaled(agg.dW1, scale)
	tBlock.WO.SubScaled(agg.dWO, scale)
	for h := 0; h < tBlock.NumHeads; h++ {
		tBlock.WQ[h].SubScaled(agg.dWQHeads[h], scale)
		tBlock.WK[h].SubScaled(agg.dWKHeads[h], scale)
		tBlock.WV[h].SubScaled(agg.dWVHeads[h], scale)
	}
	m.SubScaledByGrad(agg.dEmbedding, scale)

	return agg.loss, agg.correct, agg.tokens, agg.sortCorrect, agg.sortTokens, agg.sumCorrect, agg.sumTokens, agg.sortSamples, agg.sumSamples, nil
}

func newWorkerBatchAccum(m *embbeding.Embedding, tBlock *transformer.TransformerBlock, cW *matrix.Matrix) workerBatchAccum {
	acc := workerBatchAccum{
		dCW:        matrix.NewZeroMatrix(cW.Rows, cW.Cols),
		dW1:        matrix.NewZeroMatrix(tBlock.W1.Rows, tBlock.W1.Cols),
		dW2:        matrix.NewZeroMatrix(tBlock.W2.Rows, tBlock.W2.Cols),
		dWO:        matrix.NewZeroMatrix(tBlock.WO.Rows, tBlock.WO.Cols),
		dWQHeads:   make([]*matrix.Matrix, tBlock.NumHeads),
		dWKHeads:   make([]*matrix.Matrix, tBlock.NumHeads),
		dWVHeads:   make([]*matrix.Matrix, tBlock.NumHeads),
		dEmbedding: matrix.NewZeroMatrix(m.Vocab, m.Dimensions),
	}
	for h := 0; h < tBlock.NumHeads; h++ {
		acc.dWQHeads[h] = matrix.NewZeroMatrix(tBlock.WQ[h].Rows, tBlock.WQ[h].Cols)
		acc.dWKHeads[h] = matrix.NewZeroMatrix(tBlock.WK[h].Rows, tBlock.WK[h].Cols)
		acc.dWVHeads[h] = matrix.NewZeroMatrix(tBlock.WV[h].Rows, tBlock.WV[h].Cols)
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

func computeSampleGrad(source []uint8, target []uint8, parsed prompt.ParsedPrompt, m *embbeding.Embedding, tBlock *transformer.TransformerBlock, cW *matrix.Matrix, modelDim int) (*sampleGrad, error) {
	sequenceLen := len(source)
	sequenceMatrix := matrix.NewZeroMatrix(sequenceLen, modelDim)
	for j := range sequenceLen {
		vec := m.GetVecForEntry(source[j])
		positioned := transformer.AddPosition(j, vec)
		copy(sequenceMatrix.Vec[j], positioned.Data)
	}

	attention, mhaCache, err := transformer.MultiHeadAttention(sequenceMatrix, tBlock)
	if err != nil {
		return nil, err
	}
	out := transformer.AddAndNorm(sequenceMatrix, attention)
	ffn, ffnCache := transformer.ExpandWithCache(out, tBlock.W1, tBlock.W2)

	logits := ffn.Mul(cW)
	choices := transformer.SelectChoice(logits)

	correct, measuredTokens := scorePrediction(choices, target, parsed)

	lossValue, dLogits := lossfn.CrossEntropyWithGrad(logits, target)

	dFFN := dLogits.Mul(cW.Transpose())
	dCW := ffn.Transpose().Mul(dLogits)

	dW1, dW2, dOutFromFFN := transformer.FFNBackward(dFFN, ffnCache, tBlock.W1, tBlock.W2)
	dSequenceSkip, dAttention := transformer.AddAndNormBackward(dOutFromFFN, sequenceMatrix, attention)

	dWQHeads, dWKHeads, dWVHeads, dWO, dSequenceFromAttention, err := transformer.MultiHeadAttentionBackward(dAttention, mhaCache, tBlock)
	if err != nil {
		return nil, err
	}
	dSequence := dSequenceSkip.Add(dSequenceFromAttention)

	dEmbedding := matrix.NewZeroMatrix(m.Vocab, m.Dimensions)
	for pos := range sequenceLen {
		token := int(source[pos])
		if token < 0 || token >= m.Vocab {
			return nil, fmt.Errorf("token out of vocab: %d", token)
		}
		for j := 0; j < m.Dimensions; j++ {
			dEmbedding.Vec[token][j] += dSequence.Vec[pos][j]
		}
	}

	return &sampleGrad{
		loss:        lossValue,
		correct:     correct,
		tokens:      measuredTokens,
		task:        parsed.Task,
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

func scorePrediction(pred []int, target []uint8, parsed prompt.ParsedPrompt) (int, int) {
	correct := 0
	tokens := 0

	if parsed.Task == prompt.TaskSum {
		for _, idx := range []int{2 + prompt.MaxListLen - 2, 2 + prompt.MaxListLen - 1} {
			tokens++
			if uint8(pred[idx]) == target[idx] {
				correct++
			}
		}
		return correct, tokens
	}

	for i := 0; i < parsed.Count; i++ {
		idx := 2 + i
		tokens++
		if uint8(pred[idx]) == target[idx] {
			correct++
		}
	}

	return correct, tokens
}

func openDataset(path string) (*os.File, int64, int, error) {
	stat, err := os.Stat(path)
	if err != nil {
		return nil, 0, 0, err
	}

	size := stat.Size()
	if size%decode.RecordSize != 0 {
		return nil, 0, 0, fmt.Errorf("invalid file size: %d is not divisible by %d", size, decode.RecordSize)
	}

	totalRecords := size / decode.RecordSize
	if totalRecords == 0 {
		return nil, 0, 0, fmt.Errorf("empty dataset")
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}

	workers := max(min(runtime.NumCPU(), int(totalRecords)), 1)

	return f, totalRecords, workers, nil
}
