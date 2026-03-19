package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"time"

	"github.com/juanpablocruz/attention/gen/internal/output"
)

/***
* TASK: Create thousands of random short list of integers
* The generator needs to produce two things for every example:
* - Source(X): A sequence of random integers (eg. [8, 3, 5, 1])
* - Target(Y): The same integers sorted (eg. [1, 3, 5, 8])
*
* Constrains:
* - Vocabulary size: 0-9
* - Sequence length: 5
* - Special Tokens: <SOS> start of sentence, and <EOS> end of sentence
* - Batching: needs to be read in batches -> NO JSON
 */

func generateOutput() output.Output {
	source := [5]uint8{
		uint8(rand.Intn(10)),
		uint8(rand.Intn(10)),
		uint8(rand.Intn(10)),
		uint8(rand.Intn(10)),
		uint8(rand.Intn(10)),
	}

	var target [5]uint8
	copy(target[:], source[:])

	slices.Sort(target[:])

	return output.Output{
		Source: source,
		Target: target,
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

	var buf [10]byte
	for range n {
		out := generateOutput()
		copy(buf[0:5], out.Source[:])
		copy(buf[5:10], out.Target[:])
		if _, err = w.Write(buf[:]); err != nil {
			log.Fatal("write error:", err)
		}
	}

	log.Println("Elapsed:", time.Since(start))
	fmt.Println("gen done.")
}
