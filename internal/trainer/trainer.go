package trainer

import (
	"context"
	"time"

	"github.com/juanpablocruz/attention/gen/internal/attention"
	"github.com/juanpablocruz/attention/gen/internal/intent"
	"github.com/juanpablocruz/attention/gen/internal/matrix"
	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/internal/prompt"
)

const (
	defaultEpochs       = 3
	defaultCheckpoint   = "./checkpoints/embed_model.bin"
	defaultIntentPath   = "./checkpoints/intent_model.bin"
	defaultVocabSize    = prompt.VocabSize
	defaultModelDim     = 64
	defaultNumHeads     = 4
	defaultLearningRate = float32(0.01)
	defaultBatchSize    = 256
	envBatchSize        = "ATTN_BATCH_SIZE"
	envGradWorkers      = "ATTN_GRAD_WORKERS"
	envNumHeads         = "ATTN_NUM_HEADS"
	envModelDim         = "ATTN_MODEL_DIM"
)

type Config struct {
	DatasetPath    string
	Epochs         int
	CheckpointPath string
	IntentPath     string
	VocabSize      int
	ModelDim       int
	NumHeads       int
	LearningRate   float32
	BatchSize      int
	GradWorkers    int
	Progress       ProgressFactory
	ProgressInterval time.Duration
}

type ProgressUpdate struct {
	Loss     float64
	Accuracy float64
	Speed    float64
	ETA      time.Duration
}

type ProgressReporter interface {
	ShouldRender(now time.Time, interval time.Duration) bool
	Render(now time.Time, current int64, update ProgressUpdate)
	Finish()
}

type ProgressFactory func(epoch, totalEpochs int, total int64, start time.Time) ProgressReporter

func (c Config) CheckpointPathOrDefault() string {
	if c.CheckpointPath != "" {
		return c.CheckpointPath
	}
	return defaultCheckpoint
}

func (c Config) IntentPathOrDefault() string {
	if c.IntentPath != "" {
		return c.IntentPath
	}
	return defaultIntentPath
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
	samples []preparedSample
	packed  *attention.PackedModel
	resp    chan<- workerBatchAccum
}

type preparedSample struct {
	source [prompt.SequenceLen]uint8
	target [prompt.SequenceLen]uint8
	task   string
	count  int
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

	model := attention.NewModel(cfg.VocabSize, cfg.ModelDim, cfg.NumHeads)

	loaded, err := model.LoadCheckpoint(cfg.CheckpointPath)
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
		avgLoss, tokenAcc, seen, sortAcc, sumAcc, sortSamples, sumSamples, canceled, err := trainEpoch(ctx, epoch, cfg.Epochs, cfg.DatasetPath, model, cfg.LearningRate, cfg.BatchSize, cfg.GradWorkers, cfg.Progress, cfg.ProgressInterval)
		if err != nil {
			return loaded, nil, interrupted, err
		}
		if canceled {
			interrupted = true
		}

		if err := model.SaveCheckpoint(cfg.CheckpointPath); err != nil {
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

type IntentEpochMetrics struct {
	Epoch       int
	TotalEpochs int
	Samples     int64
	TrainLoss   float64
	TrainAcc    float64
	ValLoss     float64
	ValAcc      float64
	TrainSeen   int64
	ValSeen     int64
	Elapsed     time.Duration
}

func TrainIntent(ctx context.Context, cfg Config) (bool, []IntentEpochMetrics, bool, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	cfg = normalizeConfig(cfg)

	intentModel, err := intent.Load(cfg.IntentPath)
	loaded := err == nil && intentModel != nil
	if intentModel == nil {
		intentModel = intent.NewModel()
	}

	metrics := make([]IntentEpochMetrics, 0, cfg.Epochs)
	interrupted := false

	for epoch := 1; epoch <= cfg.Epochs; epoch++ {
		if ctx.Err() != nil {
			interrupted = true
			break
		}

		start := time.Now()
		trainLoss, trainAcc, valLoss, valAcc, trainSeen, valSeen, canceled, err := trainIntentEpoch(ctx, epoch, cfg.Epochs, cfg.DatasetPath, intentModel, cfg.Progress, cfg.ProgressInterval)
		if err != nil {
			return loaded, nil, interrupted, err
		}
		if canceled {
			interrupted = true
		}

		if err := intentModel.Save(cfg.IntentPath); err != nil {
			return loaded, nil, interrupted, err
		}

		metrics = append(metrics, IntentEpochMetrics{
			Epoch:       epoch,
			TotalEpochs: cfg.Epochs,
			Samples:     trainSeen + valSeen,
			TrainLoss:   trainLoss,
			TrainAcc:    trainAcc,
			ValLoss:     valLoss,
			ValAcc:      valAcc,
			TrainSeen:   trainSeen,
			ValSeen:     valSeen,
			Elapsed:     time.Since(start),
		})

		if canceled {
			break
		}
	}

	return loaded, metrics, interrupted, nil
}
