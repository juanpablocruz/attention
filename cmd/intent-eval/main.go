package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"sort"
	"strconv"

	"github.com/juanpablocruz/attention/gen/internal/intent"
	"github.com/juanpablocruz/attention/gen/internal/output"
)

type miss struct {
	prompt string
	got    intent.Label
	want   intent.Label
	conf   float32
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: intent-eval <dataset.bin> [max_errors]")
	}
	maxErrors := 20
	if len(os.Args) >= 3 {
		v, err := strconv.Atoi(os.Args[2])
		if err != nil || v < 1 {
			log.Fatal("invalid max_errors")
		}
		maxErrors = v
	}

	model, err := intent.Load("./checkpoints/intent_model.bin")
	if err != nil {
		log.Fatalf("could not load intent model: %v", err)
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatalf("could not open dataset: %v", err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			log.Printf("could not close dataset: %v", err)
		}
	}()

	buf := make([]byte, output.RecordSize)
	var confusion [3][3]int64
	var total int64
	var correct int64
	misses := make([]miss, 0, maxErrors*3)

	for {
		_, err := io.ReadFull(f, buf)
		if err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				break
			}
			log.Fatalf("read error: %v", err)
		}

		rec := output.DecodeRecord(buf)
		want, err := labelFromRecord(rec.Intent)
		if err != nil {
			continue
		}
		got, probs := model.Predict(rec.Prompt)
		conf := probs[int(got)]

		confusion[int(want)][int(got)]++
		total++
		if want == got {
			correct++
			continue
		}
		misses = append(misses, miss{prompt: rec.Prompt, got: got, want: want, conf: conf})
	}

	if total == 0 {
		log.Fatal("dataset has no records")
	}

	acc := float64(correct) / float64(total)
	fmt.Printf("samples=%d accuracy=%.4f\n", total, acc)
	fmt.Println("confusion (rows=true, cols=pred):")
	fmt.Printf("           asc      desc     sum\n")
	fmt.Printf("true asc   %-8d %-8d %-8d\n", confusion[0][0], confusion[0][1], confusion[0][2])
	fmt.Printf("true desc  %-8d %-8d %-8d\n", confusion[1][0], confusion[1][1], confusion[1][2])
	fmt.Printf("true sum   %-8d %-8d %-8d\n", confusion[2][0], confusion[2][1], confusion[2][2])

	sort.Slice(misses, func(i, j int) bool {
		return misses[i].conf > misses[j].conf
	})
	if len(misses) > maxErrors {
		misses = misses[:maxErrors]
	}

	fmt.Printf("hard errors (top %d by wrong-confidence):\n", len(misses))
	for i, m := range misses {
		fmt.Printf("%d) want=%s got=%s conf=%.3f prompt=%q\n", i+1, labelName(m.want), labelName(m.got), m.conf, m.prompt)
	}
}

func labelFromRecord(v uint8) (intent.Label, error) {
	switch v {
	case output.IntentSortAsc:
		return intent.SortAsc, nil
	case output.IntentSortDesc:
		return intent.SortDesc, nil
	case output.IntentSum:
		return intent.Sum, nil
	default:
		return intent.SortAsc, fmt.Errorf("invalid intent label: %d", v)
	}
}

func labelName(l intent.Label) string {
	switch l {
	case intent.SortAsc:
		return "asc"
	case intent.SortDesc:
		return "desc"
	case intent.Sum:
		return "sum"
	default:
		return "unknown"
	}
}
