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
		log.Fatal("usage: intent <train_dataset.bin>")
	}

	cfg := trainer.Config{
		DatasetPath: os.Args[1],
		Progress: func(epoch, totalEpochs int, total int64, start time.Time) trainer.ProgressReporter {
			return progress.NewReporter(epoch, totalEpochs, total, start, formatIntentProgress)
		},
		ProgressInterval: 200 * time.Millisecond,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	loaded, metrics, interrupted, err := trainer.TrainIntent(ctx, cfg)
	if err != nil {
		log.Fatalf("intent training failed: %v", err)
	}

	if loaded {
		log.Printf("loaded intent checkpoint: %s", cfg.IntentPathOrDefault())
	} else {
		log.Printf("starting intent model from random initialization")
	}

	for _, m := range metrics {
		fmt.Printf("epoch=%d/%d samples=%d train_samples=%d train_loss=%.6f train_acc=%.4f val_samples=%d val_loss=%.6f val_acc=%.4f elapsed=%s\n", m.Epoch, m.TotalEpochs, m.Samples, m.TrainSeen, m.TrainLoss, m.TrainAcc, m.ValSeen, m.ValLoss, m.ValAcc, m.Elapsed)
	}

	if interrupted {
		log.Printf("intent training interrupted, latest checkpoint saved")
	}
}

func formatIntentProgress(update trainer.ProgressUpdate) string {
	return fmt.Sprintf(
		"loss=%.6f acc=%s speed=%.0f/s eta=%s",
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
		return color.New(color.FgGreen).Sprintf("%.4f", acc)
	}
}
