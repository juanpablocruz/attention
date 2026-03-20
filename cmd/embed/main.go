package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fatih/color"
	"github.com/juanpablocruz/attention/gen/internal/progress"
	"github.com/juanpablocruz/attention/gen/internal/trainer"
)

func main() {
	if len(os.Args) != 2 {
		log.Fatal("usage: embed <train_dataset.bin>")
	}

	cfg := trainer.Config{
		DatasetPath: os.Args[1],
		Progress: func(epoch, totalEpochs int, total int64, start time.Time) trainer.ProgressReporter {
			return progress.NewReporter(epoch, totalEpochs, total, start, formatEmbedProgress)
		},
		ProgressInterval: 200 * time.Millisecond,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	loaded, epochMetrics, interrupted, err := trainer.Train(ctx, cfg)
	if err != nil {
		log.Fatalf("training failed: %v", err)
	}

	if loaded {
		log.Printf("loaded checkpoint: %s", cfg.CheckpointPathOrDefault())
	} else {
		log.Printf("starting from random initialization")
	}

	for _, m := range epochMetrics {
		fmt.Printf("epoch=%d/%d train_samples=%d train_loss=%.6f train_token_acc=%.4f sort_token_acc=%.4f sort_samples=%d sum_token_acc=%.4f sum_samples=%d elapsed=%s\n", m.Epoch, m.TotalEpochs, m.Samples, m.Loss, m.TokenAcc, m.SortTokenAcc, m.SortSamples, m.SumTokenAcc, m.SumSamples, m.Elapsed)
	}

	if interrupted {
		log.Printf("training interrupted, latest checkpoint saved")
	}
}

func formatEmbedProgress(update trainer.ProgressUpdate) string {
	return fmt.Sprintf(
		"loss=%.6f token_acc=%s speed=%.0f/s eta=%s",
		update.Loss,
		colorizeAccuracy(update.Accuracy),
		update.Speed,
		update.ETA,
	)
}

func colorizeAccuracy(acc float64) string {
	switch {
	case acc < 0.4:
		return color.New(color.FgRed).Sprintf("%.4f", acc)
	case acc < 0.8:
		return color.New(color.FgYellow).Sprintf("%.4f", acc)
	default:
		return color.GreenString("%.4f", acc)
	}
}
