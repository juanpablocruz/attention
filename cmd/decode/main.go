package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/pkg/decode"
)

func main() {

	if len(os.Args) < 2 {
		log.Fatal("usage: decode <dataset.bin> [first_n]")
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

	if len(os.Args) >= 3 {
		n, err := strconv.ParseInt(os.Args[2], 10, 64)
		if err != nil || n <= 0 {
			log.Fatal("invalid first_n value")
		}
		if n > totalRecords {
			n = totalRecords
		}

		if err := printFirstN(f, n); err != nil {
			log.Fatalf("could not print first %d records: %v", n, err)
		}
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

func printFirstN(f *os.File, n int64) error {
	buf := make([]byte, decode.RecordSize)
	for i := range n {
		offset := i * decode.RecordSize
		_, err := f.ReadAt(buf, offset)
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		rec := output.DecodeRecord(buf)
		fmt.Printf("[%d]\n", i)
		fmt.Printf("  prompt: %s\n", rec.Prompt)
		fmt.Printf("  target: %s\n", rec.Target)
	}
	return nil
}
