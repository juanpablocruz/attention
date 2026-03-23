package trainer

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/juanpablocruz/attention/gen/internal/intent"
	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/pkg/decode"
)

func trainIntentEpoch(ctx context.Context, epoch, totalEpochs int, datasetPath string, intentModel *intent.Model, progressFactory ProgressFactory, progressInterval time.Duration) (_ float64, _ float64, _ float64, _ float64, trainSeen int64, valSeen int64, _ bool, err error) {
	f, totalRecords, workers, err := openDataset(datasetPath)
	if err != nil {
		return 0, 0, 0, 0, 0, 0, false, err
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

	totalLoss := 0.0
	var correct int64
	valLoss := 0.0
	var valCorrect int64
	var processed int64
	start := time.Now()
	var reporter ProgressReporter
	if progressFactory != nil {
		reporter = progressFactory(epoch, totalEpochs, totalRecords, start)
	}
	if reporter != nil {
		defer reporter.Finish()
	}

	for out := range outs {
		if ctx.Err() != nil {
			return 0, 0, 0, 0, trainSeen, valSeen, true, nil
		}

		label, err := labelFromRecord(out.Intent)
		if err != nil {
			return 0, 0, 0, 0, trainSeen, valSeen, false, err
		}

		pred, probs := intentModel.Predict(out.Prompt)
		p := float64(probs[int(label)])
		if p < 1e-9 {
			p = 1e-9
		}

		if isIntentValidationSample(out.Prompt) {
			if pred == label {
				valCorrect++
			}
			valLoss += -math.Log(p)
			valSeen++
		} else {
			if pred == label {
				correct++
			}
			totalLoss += -math.Log(p)
			intentModel.TrainStep(out.Prompt, label, 0.01)
			trainSeen++
		}
		processed++

		now := time.Now()
		if reporter != nil && progressInterval > 0 && reporter.ShouldRender(now, progressInterval) {
			avgLoss := totalLoss / float64(max(trainSeen, 1))
			acc := float64(correct) / float64(max(trainSeen, 1))
			elapsed := now.Sub(start)
			rate := float64(processed) / elapsed.Seconds()
			remaining := float64(totalRecords-processed) / max(rate, 1e-9)
			eta := time.Duration(remaining * float64(time.Second))
			reporter.Render(now, processed, ProgressUpdate{
				Loss:     avgLoss,
				Accuracy: acc,
				Speed:    rate,
				ETA:      eta.Round(time.Second),
			})
		}
	}

	if trainSeen == 0 {
		return 0, 0, 0, 0, trainSeen, valSeen, false, fmt.Errorf("dataset has no training records")
	}

	avgValLoss := 0.0
	avgValAcc := 0.0
	if valSeen > 0 {
		avgValLoss = valLoss / float64(valSeen)
		avgValAcc = float64(valCorrect) / float64(valSeen)
	}

	return totalLoss / float64(trainSeen), float64(correct) / float64(trainSeen), avgValLoss, avgValAcc, trainSeen, valSeen, false, nil
}
