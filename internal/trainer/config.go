package trainer

import (
	"os"
	"runtime"
	"strconv"
)

func normalizeConfig(cfg Config) Config {
	if cfg.Epochs <= 0 {
		cfg.Epochs = defaultEpochs
	}
	if cfg.CheckpointPath == "" {
		cfg.CheckpointPath = defaultCheckpoint
	}
	if cfg.IntentPath == "" {
		cfg.IntentPath = defaultIntentPath
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

