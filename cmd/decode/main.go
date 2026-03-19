package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/juanpablocruz/attention/gen/pkg/decode"
)

func main() {

	if len(os.Args) < 2 {
		panic("Invalid usage: gen [len of dataset]")
	}

	arg := os.Args[1]
	stat, err := os.Stat(arg)
	if err != nil {
		log.Fatalf("File %s does not exist", arg)
	}

	f, err := os.Open(arg)

	if err != nil {
		log.Fatal("Could not open file")
	}

	defer f.Close()

	var size = stat.Size()

	if size%decode.RecordSize != 0 {
		log.Fatalf("invalid file size: %d is not divisible by %d", size, decode.RecordSize)
	}

	totalRecords := size / decode.RecordSize
	if totalRecords == 0 {
		return
	}

	workers := min(int64(runtime.NumCPU()), totalRecords)

	startTime := time.Now()
	var wg sync.WaitGroup
	wg.Add(int(workers))

	decode.DecodeOutput(context.Background(), workers, totalRecords, &wg, f, nil)

	wg.Wait()
	fmt.Println("total time:", time.Since(startTime))
	fmt.Printf("decoded %d records with %d workers\n", totalRecords, workers)
}
