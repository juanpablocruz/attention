package trainer

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/juanpablocruz/attention/gen/internal/checkpoint"
	"github.com/juanpablocruz/attention/gen/internal/embbeding"
	lossfn "github.com/juanpablocruz/attention/gen/internal/loss"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/internal/transformer"
	"github.com/juanpablocruz/attention/gen/pkg/decode"
)

const (
	defaultEpochs       = 3
	defaultCheckpoint   = "./checkpoints/embed_model.bin"
	defaultVocabSize    = 10
	defaultModelDim     = 32
	defaultLearningRate = float32(0.01)
	defaultBatchSize    = 64
)

type Config struct {
	DatasetPath    string
	Epochs         int
	CheckpointPath string
	VocabSize      int
	ModelDim       int
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
	Epoch       int
	TotalEpochs int
	Samples     int64
	Loss        float64
	TokenAcc    float64
	Elapsed     time.Duration
}

type outputSink struct {
	ch chan<- output.Output
}

type sampleGrad struct {
	loss       float32
	correct    int
	tokens     int
	dCW        *matrix.Matrix
	dW1        *matrix.Matrix
	dW2        *matrix.Matrix
	dWQ        *matrix.Matrix
	dWK        *matrix.Matrix
	dWV        *matrix.Matrix
	dEmbedding *matrix.Matrix
}

func (s *outputSink) Process(o output.Output) {
	s.ch <- o
}

func Train(cfg Config) (bool, []EpochMetrics, error) {
	cfg = normalizeConfig(cfg)

	m := embbeding.New(cfg.VocabSize, cfg.ModelDim)
	tBlock := transformer.New(cfg.ModelDim)
	cW := matrix.New(cfg.ModelDim, cfg.VocabSize)

	loaded, err := checkpoint.Load(cfg.CheckpointPath, m, tBlock, cW)
	if err != nil {
		return false, nil, err
	}

	metrics := make([]EpochMetrics, 0, cfg.Epochs)
	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		start := time.Now()
		avgLoss, tokenAcc, seen, err := trainEpoch(cfg.DatasetPath, m, tBlock, cW, cfg.LearningRate, cfg.ModelDim, cfg.BatchSize, cfg.GradWorkers)
		if err != nil {
			return loaded, nil, err
		}

		if err := checkpoint.Save(cfg.CheckpointPath, m, tBlock, cW); err != nil {
			return loaded, nil, err
		}

		metrics = append(metrics, EpochMetrics{
			Epoch:       epoch,
			TotalEpochs: cfg.Epochs,
			Samples:     seen,
			Loss:        avgLoss,
			TokenAcc:    tokenAcc,
			Elapsed:     time.Since(start),
		})
	}

	return loaded, metrics, nil
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
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = defaultBatchSize
	}
	if cfg.GradWorkers <= 0 {
		cfg.GradWorkers = runtime.NumCPU()
	}
	if cfg.GradWorkers < 1 {
		cfg.GradWorkers = 1
	}
	return cfg
}

func trainEpoch(datasetPath string, m *embbeding.Embedding, tBlock *transformer.TransformerBlock, cW *matrix.Matrix, learningRate float32, modelDim, batchSize, gradWorkers int) (float64, float64, int64, error) {
	f, totalRecords, workers, err := openDataset(datasetPath)
	if err != nil {
		return 0, 0, 0, err
	}
	defer f.Close()

	outs := make(chan output.Output, 4096)
	proc := &outputSink{ch: outs}

	var wg sync.WaitGroup
	wg.Add(workers)
	decode.DecodeOutput(int64(workers), totalRecords, &wg, f, proc)

	go func() {
		wg.Wait()
		close(outs)
	}()

	var totalLoss float64
	var totalCorrect int64
	var totalTokens int64
	var seen int64
	batch := make([]output.Output, 0, batchSize)

	for out := range outs {
		batch = append(batch, out)
		if len(batch) < batchSize {
			continue
		}

		batchLoss, batchCorrect, batchTokens, err := trainBatch(batch, m, tBlock, cW, learningRate, modelDim, gradWorkers)
		if err != nil {
			return 0, 0, 0, err
		}

		totalLoss += batchLoss
		totalCorrect += batchCorrect
		totalTokens += batchTokens
		seen += int64(len(batch))
		batch = batch[:0]
	}

	if len(batch) > 0 {
		batchLoss, batchCorrect, batchTokens, err := trainBatch(batch, m, tBlock, cW, learningRate, modelDim, gradWorkers)
		if err != nil {
			return 0, 0, 0, err
		}

		totalLoss += batchLoss
		totalCorrect += batchCorrect
		totalTokens += batchTokens
		seen += int64(len(batch))
	}

	if seen == 0 {
		return 0, 0, 0, fmt.Errorf("dataset has no records")
	}

	return totalLoss / float64(seen), float64(totalCorrect) / float64(totalTokens), seen, nil
}

func trainBatch(batch []output.Output, m *embbeding.Embedding, tBlock *transformer.TransformerBlock, cW *matrix.Matrix, learningRate float32, modelDim, gradWorkers int) (float64, int64, int64, error) {
	workerCount := max(min(gradWorkers, len(batch)), 1)

	jobs := make(chan output.Output, len(batch))
	results := make(chan *sampleGrad, len(batch))
	errCh := make(chan error, 1)

	var wg sync.WaitGroup
	wg.Add(workerCount)
	for range workerCount {
		go func() {
			defer wg.Done()
			for out := range jobs {
				grad, err := computeSampleGrad(out.Source[:], out.Target[:], m, tBlock, cW, modelDim)
				if err != nil {
					select {
					case errCh <- err:
					default:
					}
					continue
				}
				results <- grad
			}
		}()
	}

	for _, out := range batch {
		jobs <- out
	}
	close(jobs)

	wg.Wait()
	close(results)

	select {
	case err := <-errCh:
		return 0, 0, 0, err
	default:
	}

	aggCW := matrix.NewZeroMatrix(cW.Rows, cW.Cols)
	aggW1 := matrix.NewZeroMatrix(tBlock.W1.Rows, tBlock.W1.Cols)
	aggW2 := matrix.NewZeroMatrix(tBlock.W2.Rows, tBlock.W2.Cols)
	aggWQ := matrix.NewZeroMatrix(tBlock.WQ.Rows, tBlock.WQ.Cols)
	aggWK := matrix.NewZeroMatrix(tBlock.WK.Rows, tBlock.WK.Cols)
	aggWV := matrix.NewZeroMatrix(tBlock.WV.Rows, tBlock.WV.Cols)
	aggEmbedding := matrix.NewZeroMatrix(m.Vocab, m.Dimensions)

	var batchLoss float64
	var batchCorrect int64
	var batchTokens int64
	count := 0

	for grad := range results {
		batchLoss += float64(grad.loss)
		batchCorrect += int64(grad.correct)
		batchTokens += int64(grad.tokens)
		count++

		aggCW.AddInPlace(grad.dCW)
		aggW1.AddInPlace(grad.dW1)
		aggW2.AddInPlace(grad.dW2)
		aggWQ.AddInPlace(grad.dWQ)
		aggWK.AddInPlace(grad.dWK)
		aggWV.AddInPlace(grad.dWV)
		aggEmbedding.AddInPlace(grad.dEmbedding)
	}

	if count == 0 {
		return 0, 0, 0, fmt.Errorf("empty gradient batch")
	}

	scale := learningRate / float32(count)
	cW.SubScaled(aggCW, scale)
	tBlock.W2.SubScaled(aggW2, scale)
	tBlock.W1.SubScaled(aggW1, scale)
	tBlock.WQ.SubScaled(aggWQ, scale)
	tBlock.WK.SubScaled(aggWK, scale)
	tBlock.WV.SubScaled(aggWV, scale)
	m.SubScaledByGrad(aggEmbedding, scale)

	return batchLoss, batchCorrect, batchTokens, nil
}

func computeSampleGrad(source []uint8, target []uint8, m *embbeding.Embedding, tBlock *transformer.TransformerBlock, cW *matrix.Matrix, modelDim int) (*sampleGrad, error) {
	sequenceLen := len(source)
	sequenceMatrix := matrix.NewZeroMatrix(sequenceLen, modelDim)
	for j := range sequenceLen {
		vec := m.GetVecForEntry(source[j])
		positioned := transformer.AddPosition(j, vec)
		copy(sequenceMatrix.Vec[j], positioned.Data)
	}

	q := sequenceMatrix.Mul(tBlock.WQ)
	k := sequenceMatrix.Mul(tBlock.WK)
	v := sequenceMatrix.Mul(tBlock.WV)

	attention := transformer.ScaledAttention(q, k, v)
	out := transformer.AddAndNorm(sequenceMatrix, attention)
	ffn, ffnCache := transformer.ExpandWithCache(out, tBlock.W1, tBlock.W2)

	logits := ffn.Mul(cW)
	choices := transformer.SelectChoice(logits)

	correct := 0
	for i := range sequenceLen {
		if uint8(choices[i]) == target[i] {
			correct++
		}
	}

	lossValue, dLogits := lossfn.CrossEntropyWithGrad(logits, target)

	dFFN := dLogits.Mul(cW.Transpose())
	dCW := ffn.Transpose().Mul(dLogits)

	dW1, dW2, dOutFromFFN := transformer.FFNBackward(dFFN, ffnCache, tBlock.W1, tBlock.W2)
	dSequenceSkip, dAttention := transformer.AddAndNormBackward(dOutFromFFN, sequenceMatrix, attention)

	dQ, dK, dV := transformer.ScaledAttentionBackward(dAttention, q, k, v)
	dWQ := sequenceMatrix.Transpose().Mul(dQ)
	dWK := sequenceMatrix.Transpose().Mul(dK)
	dWV := sequenceMatrix.Transpose().Mul(dV)

	dSequenceQ := dQ.Mul(tBlock.WQ.Transpose())
	dSequenceK := dK.Mul(tBlock.WK.Transpose())
	dSequenceV := dV.Mul(tBlock.WV.Transpose())
	dSequenceFromAttention := dSequenceQ.Add(dSequenceK).Add(dSequenceV)
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
		loss:       lossValue,
		correct:    correct,
		tokens:     sequenceLen,
		dCW:        dCW,
		dW1:        dW1,
		dW2:        dW2,
		dWQ:        dWQ,
		dWK:        dWK,
		dWV:        dWV,
		dEmbedding: dEmbedding,
	}, nil
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
