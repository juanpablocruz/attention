package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/internal/prompt"
)

/***
* TASK: Create random short list examples with an operation text.
* The generator produces semantic records with:
* - Source list: [n1, n2, n3, n4, n5]
* - Target list: sorted source by operation
* - Operation: "asc" or "desc"
 */

func generateOutput() output.Output {
	numbers := [5]uint8{
		uint8(rand.Intn(10)),
		uint8(rand.Intn(10)),
		uint8(rand.Intn(10)),
		uint8(rand.Intn(10)),
		uint8(rand.Intn(10)),
	}
	order := "asc"
	if rand.Intn(2) == 1 {
		order = "desc"
	}
	p := prompt.BuildPrompt(numbers, order)
	t := prompt.BuildTarget(numbers, order)

	return output.Output{
		Prompt: p,
		Target: t,
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

	for range n {
		out := generateOutput()
		buf := out.EncodeRecord()
		if _, err = w.Write(buf[:]); err != nil {
			log.Fatal("write error:", err)
		}
	}

	log.Println("Elapsed:", time.Since(start))
	fmt.Println("gen done.")
}
