package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/internal/prompt"
)

var sortAscTemplates = []string{
	"sort list [%s] asc",
	"sort this list [%s] in ascending order",
	"order the numbers %s from smallest to largest",
}

var sortDescTemplates = []string{
	"sort list [%s] desc",
	"sort this list [%s] in descending order",
	"order the numbers %s from largest to smallest",
}

var sumTemplates = []string{
	"sum the list [%s]",
	"add these numbers %s",
	"what is the total of list [%s]",
}

func generateOutput(rng *rand.Rand) output.Output {
	listLen := prompt.MinListLen + rng.Intn(prompt.MaxListLen-prompt.MinListLen+1)
	numbers := make([]uint8, listLen)
	values := make([]string, 0, listLen)
	for i := range listLen {
		numbers[i] = uint8(rng.Intn(10))
		values = append(values, strconv.Itoa(int(numbers[i])))
	}
	commaJoined := strings.Join(values, ",")
	spaceJoined := strings.Join(values, " ")

	taskRoll := rng.Intn(3) // 0,1 => sort, 2 => sum
	task := prompt.TaskSort
	order := "asc"
	promptText := ""

	if taskRoll == 2 {
		task = prompt.TaskSum
		template := sumTemplates[rng.Intn(len(sumTemplates))]
		if strings.Contains(template, "[%s]") {
			promptText = fmt.Sprintf(template, commaJoined)
		} else {
			promptText = fmt.Sprintf(template, spaceJoined)
		}
	} else {
		if rng.Intn(2) == 1 {
			order = "desc"
			template := sortDescTemplates[rng.Intn(len(sortDescTemplates))]
			if strings.Contains(template, "[%s]") {
				promptText = fmt.Sprintf(template, commaJoined)
			} else {
				promptText = fmt.Sprintf(template, spaceJoined)
			}
		} else {
			template := sortAscTemplates[rng.Intn(len(sortAscTemplates))]
			if strings.Contains(template, "[%s]") {
				promptText = fmt.Sprintf(template, commaJoined)
			} else {
				promptText = fmt.Sprintf(template, spaceJoined)
			}
		}
	}

	targetText := prompt.BuildTarget(task, numbers, order)

	return output.Output{
		Prompt: promptText,
		Target: targetText,
	}
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: gen [len of dataset]")
	}

	arg := os.Args[1]
	n, err := strconv.ParseInt(arg, 10, 64)
	if err != nil {
		log.Fatal("invalid number")
	}

	if err := os.MkdirAll("./dataset", os.ModePerm); err != nil {
		log.Fatal("could not create dataset folder")
	}

	filename := fmt.Sprintf("./dataset/%d.bin", time.Now().Unix())
	f, err := os.Create(filename)
	if err != nil {
		log.Fatalf("could not create file %s: %v", filename, err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 4*1024*1024)
	defer w.Flush()

	start := time.Now()
	log.Println("Start:", start)

	workers := runtime.NumCPU()
	if int64(workers) > n {
		workers = int(n)
	}
	if workers < 1 {
		workers = 1
	}

	recordBatches := make(chan []byte, workers*2)
	errCh := make(chan error, 1)

	var wg sync.WaitGroup
	wg.Add(workers)
	for wid := 0; wid < workers; wid++ {
		count := n / int64(workers)
		if int64(wid) < n%int64(workers) {
			count++
		}

		go func(workerID int, count int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(workerID)*7919))
			const batchRecords int64 = 4096

			remaining := count
			for remaining > 0 {
				nrecs := min(remaining, batchRecords)

				buf := make([]byte, int(nrecs)*output.RecordSize)
				offset := 0
				for range nrecs {
					out := generateOutput(rng)
					rec := out.EncodeRecord()
					copy(buf[offset:offset+output.RecordSize], rec[:])
					offset += output.RecordSize
				}

				select {
				case recordBatches <- buf:
				default:
					recordBatches <- buf
				}

				remaining -= nrecs
			}
		}(wid, count)
	}

	go func() {
		wg.Wait()
		close(recordBatches)
	}()

	for batch := range recordBatches {
		if _, err = w.Write(batch); err != nil {
			select {
			case errCh <- err:
			default:
			}
			break
		}
	}

	select {
	case e := <-errCh:
		log.Fatal("write error:", e)
	default:
	}

	log.Println("Elapsed:", time.Since(start))
	fmt.Println("gen done.")
}
