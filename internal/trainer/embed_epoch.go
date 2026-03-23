package trainer

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/juanpablocruz/attention/gen/internal/attention"
	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/pkg/decode"
)

func trainEpoch(ctx context.Context, epoch, totalEpochs int, datasetPath string, model *attention.Model, learningRate float32, batchSize, gradWorkers int, progressFactory ProgressFactory, progressInterval time.Duration) (_ float64, _ float64, _ int64, _ float64, _ float64, _ int64, _ int64, _ bool, err error) {
	f, totalRecords, workers, err := openDataset(datasetPath)
	if err != nil {
		return 0, 0, 0, 0, 0, 0, 0, false, err
	}
	defer func() {
		err = errors.Join(err, f.Close())
	}()

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

					accum := newWorkerBatchAccum(model)
					for _, sample := range work.samples {
						if ctx.Err() != nil {
							accum.err = ctx.Err()
							break
						}

						grad, err := computeSampleGrad(sample.source[:], sample.target[:], sample.task, sample.count, model, work.packed)
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

	start := time.Now()
	var reporter ProgressReporter
	if progressFactory != nil {
		reporter = progressFactory(epoch, totalEpochs, totalRecords, start)
	}
	if reporter != nil {
		defer reporter.Finish()
	}

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

		batchLoss, batchCorrect, batchTokens, aggSortCorrect, aggSortTokens, aggSumCorrect, aggSumTokens, aggSortSamples, aggSumSamples, err := trainBatch(ctx, batch, model, learningRate, gradWorkers, workCh)
		if err != nil {
			if errors.Is(err, context.Canceled) {
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
		if reporter != nil && progressInterval > 0 && reporter.ShouldRender(now, progressInterval) {
			avgLoss := totalLoss / float64(max(seen, 1))
			tokenAcc := float64(totalCorrect) / float64(max(totalTokens, 1))
			elapsed := now.Sub(start)
			rate := float64(seen) / elapsed.Seconds()
			remaining := float64(totalRecords-seen) / max(rate, 1e-9)
			eta := time.Duration(remaining * float64(time.Second))
			reporter.Render(now, seen, ProgressUpdate{
				Loss:     avgLoss,
				Accuracy: tokenAcc,
				Speed:    rate,
				ETA:      eta.Round(time.Second),
			})
		}
	}

	if len(batch) > 0 {
		batchLoss, batchCorrect, batchTokens, aggSortCorrect, aggSortTokens, aggSumCorrect, aggSumTokens, aggSortSamples, aggSumSamples, err := trainBatch(ctx, batch, model, learningRate, gradWorkers, workCh)
		if err != nil {
			if errors.Is(err, context.Canceled) {
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
