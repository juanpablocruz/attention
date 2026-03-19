package decode

import (
	"io"
	"log"
	"sync"

	"github.com/juanpablocruz/attention/gen/internal/output"
)

const RecordSize = 10

type OutputProcesser interface {
	Process(output.Output)
}

func DecodeOutput(workers, totalRecords int64, wg *sync.WaitGroup, f io.ReaderAt, process OutputProcesser) {

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
