package decode

import (
	"context"
	"io"
	"log"
	"sync"

	"github.com/juanpablocruz/attention/gen/internal/output"
)

const RecordSize = output.RecordSize

type OutputProcesser interface {
	Process(output.Output)
}

func DecodeOutput(ctx context.Context, workers, totalRecords int64, wg *sync.WaitGroup, f io.ReaderAt, process OutputProcesser) {
	if ctx == nil {
		ctx = context.Background()
	}

	recordsPerWorker := totalRecords / workers
	for w := range workers {
		startRecord := w * recordsPerWorker
		endRecord := startRecord + recordsPerWorker

		if w == workers-1 {
			endRecord = totalRecords
		}

		go func(workerID, startRecord, endRecord int64) {
			defer wg.Done()

			const batchRecords = 4096
			buf := make([]byte, batchRecords*RecordSize)

			for current := startRecord; current < endRecord; {
				select {
				case <-ctx.Done():
					return
				default:
				}

				remaining := endRecord - current

				nrecs := min(remaining, int64(batchRecords))

				nbytes := nrecs * RecordSize
				chunk := buf[:nbytes]

				offset := current * RecordSize

				_, err := f.ReadAt(chunk, offset)
				if err != nil && err != io.EOF {
					log.Fatalf("worker %d read error at offset %d: %v", workerID, offset, err)
				}

				for i := range nrecs {
					select {
					case <-ctx.Done():
						return
					default:
					}

					start := i * RecordSize
					end := start + RecordSize
					out := output.DecodeRecord(chunk[start:end])
					if process != nil {
						process.Process(out)
					}
				}

				current += nrecs
			}
		}(w, startRecord, endRecord)
	}
}
