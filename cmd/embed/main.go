package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/juanpablocruz/attention/gen/internal/trainer"
)

func main() {
	if len(os.Args) != 2 {
		log.Fatal("usage: embed <train_dataset.bin>")
	}

	cfg := trainer.Config{
		DatasetPath: os.Args[1],
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
		fmt.Printf("epoch=%d/%d train_samples=%d train_loss=%.6f train_token_acc=%.4f elapsed=%s\n", m.Epoch, m.TotalEpochs, m.Samples, m.Loss, m.TokenAcc, m.Elapsed)
	}

	if interrupted {
		log.Printf("training interrupted, latest checkpoint saved")
	}
}
