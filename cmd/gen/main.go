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
	"unicode"

	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/internal/prompt"
)

type genMode string

const (
	modeOps    genMode = "ops"
	modeIntent genMode = "intent"
	modeOOD    genMode = "ood"
)

type lexicon struct {
	sortVerbs   []string
	sortOpeners []string
	listNouns   []string
	ascPhrases  []string
	descPhrases []string
	sumVerbs    []string
	sumObjects  []string
	sumAsk      []string
}

var sortAscTemplates = []string{
	"sort list [%s] asc",
	"sort this list [%s] in ascending order",
	"order the numbers %s from smallest to largest",
	"get this list from lowest to highest [%s]",
	"arrange [%s] from low to high",
}

var sortDescTemplates = []string{
	"sort list [%s] desc",
	"sort this list [%s] in descending order",
	"order the numbers %s from largest to smallest",
	"get this list from highest to lowest [%s]",
	"arrange [%s] from high to low",
	"get this list from higest to lowest [%s]",
}

var sumTemplates = []string{
	"sum the list [%s]",
	"add these numbers %s",
	"what is the total of list [%s]",
}

var trainLexicon = lexicon{
	sortVerbs:   []string{"sort", "order", "arrange", "organize", "rank", "line up", "reorder"},
	sortOpeners: []string{"", "please", "can you", "could you", "would you", "i need you to", "help me"},
	listNouns:   []string{"list", "array", "numbers", "values", "digits", "sequence", "set"},
	ascPhrases: []string{
		"ascending order", "from smallest to largest", "from lowest to highest", "from low to high", "least to greatest",
		"increasing order", "non-descending order", "not in descending order", "opposite of descending order", "anything but descending order",
	},
	descPhrases: []string{
		"descending order", "from largest to smallest", "from highest to lowest", "from high to low", "greatest to least",
		"decreasing order", "non-ascending order", "not in ascending order", "opposite of ascending order", "anything but ascending order",
		"from higest to lowest", "non ascendent order",
	},
	sumVerbs:   []string{"sum", "add", "total", "calculate", "compute", "find"},
	sumObjects: []string{"the numbers", "this list", "this array", "these values", "these digits", "this sequence"},
	sumAsk:     []string{"what is the total", "what's the sum", "give me the total", "give me the sum", "how much is"},
}

var oodLexicon = lexicon{
	sortVerbs:   []string{"classify", "place", "stack", "sequence", "resequence", "sort out"},
	sortOpeners: []string{"hey", "quickly", "for me", "assistant", "kindly", "without delay"},
	listNouns:   []string{"inputs", "items", "entries", "figures", "integers", "digits"},
	ascPhrases: []string{
		"in non-descendent fashion", "in not-descending sequence", "with minima first", "small to big", "upward numeric order",
		"reverse-descending", "counter-descending direction", "the non descendent order", "other than descending", "not descending",
	},
	descPhrases: []string{
		"in non-ascendent fashion", "in not-ascending sequence", "with maxima first", "big to small", "downward numeric order",
		"reverse-ascending", "counter-ascending direction", "the non ascendent order", "other than ascending", "not ascending",
	},
	sumVerbs:   []string{"accumulate", "aggregate", "combine", "count up", "roll up", "compute total"},
	sumObjects: []string{"those integers", "that sequence", "the values shown", "these entries", "the listed numbers"},
	sumAsk:     []string{"return the aggregate", "what total do they make", "how much do they add to", "what is their combined value"},
}

func main() {
	mode, count := parseArgs(os.Args[1:])
	filename := datasetFilename(mode)
	if err := writeDataset(filename, count, mode); err != nil {
		log.Fatal(err)
	}
	fmt.Println("gen done:", filename)
}

func parseArgs(args []string) (genMode, int64) {
	if len(args) != 2 {
		log.Fatal("usage: gen <ops|intent|ood> <len>")
	}

	mode := genMode(strings.ToLower(strings.TrimSpace(args[0])))
	switch mode {
	case modeOps, modeIntent, modeOOD:
	default:
		log.Fatal("invalid mode, use ops, intent, or ood")
	}

	n, err := strconv.ParseInt(args[1], 10, 64)
	if err != nil || n < 1 {
		log.Fatal("invalid dataset length")
	}

	return mode, n
}

func datasetFilename(mode genMode) string {
	now := time.Now().Unix()
	switch mode {
	case modeOps:
		return fmt.Sprintf("./dataset/%d.bin", now)
	case modeIntent:
		return fmt.Sprintf("./dataset/intent_train_%d.bin", now)
	case modeOOD:
		return fmt.Sprintf("./dataset/intent_ood_%d.bin", now)
	default:
		return fmt.Sprintf("./dataset/%s_%d.bin", mode, now)
	}
}

func writeDataset(filename string, n int64, mode genMode) error {
	if err := os.MkdirAll("./dataset", os.ModePerm); err != nil {
		return fmt.Errorf("could not create dataset folder: %w", err)
	}

	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("could not create file %s: %w", filename, err)
	}

	w := bufio.NewWriterSize(f, 4*1024*1024)
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
					out := generateOutput(rng, mode)
					rec := out.EncodeRecord()
					copy(buf[offset:offset+output.RecordSize], rec[:])
					offset += output.RecordSize
				}
				recordBatches <- buf
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
	case writeErr := <-errCh:
		return writeErr
	default:
	}

	if err := w.Flush(); err != nil {
		return fmt.Errorf("flush %s: %w", filename, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close %s: %w", filename, err)
	}

	log.Println("Elapsed:", time.Since(start))
	return nil
}

func generateOutput(rng *rand.Rand, mode genMode) output.Output {
	if mode == modeOps {
		return generateOpsOutput(rng)
	}
	return generateIntentOutput(rng, mode)
}

func generateOpsOutput(rng *rand.Rand) output.Output {
	listLen := prompt.MinListLen + rng.Intn(prompt.MaxListLen-prompt.MinListLen+1)
	numbers := make([]uint8, listLen)
	values := make([]string, 0, listLen)
	for i := range listLen {
		numbers[i] = uint8(rng.Intn(10))
		values = append(values, strconv.Itoa(int(numbers[i])))
	}
	commaJoined := strings.Join(values, ",")
	spaceJoined := strings.Join(values, " ")

	taskRoll := rng.Intn(4)
	task := prompt.TaskSort
	order := "asc"
	promptText := ""
	intentLabel := uint8(output.IntentSortAsc)

	if taskRoll >= 2 {
		task = prompt.TaskSum
		intentLabel = uint8(output.IntentSum)
		template := sumTemplates[rng.Intn(len(sumTemplates))]
		if strings.Contains(template, "[%s]") {
			promptText = fmt.Sprintf(template, commaJoined)
		} else {
			promptText = fmt.Sprintf(template, spaceJoined)
		}
	} else if rng.Intn(2) == 1 {
		order = "desc"
		intentLabel = uint8(output.IntentSortDesc)
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

	targetText := prompt.BuildTarget(task, numbers, order)
	return output.Output{
		Prompt: promptText,
		Target: targetText,
		Intent: intentLabel,
	}
}

func generateIntentOutput(rng *rand.Rand, mode genMode) output.Output {
	lex := trainLexicon
	if mode == modeOOD {
		lex = oodLexicon
	}

	listLen := prompt.MinListLen + rng.Intn(prompt.MaxListLen-prompt.MinListLen+1)
	numbers := make([]uint8, listLen)
	values := make([]string, 0, listLen)
	for i := range listLen {
		numbers[i] = uint8(rng.Intn(10))
		values = append(values, strconv.Itoa(int(numbers[i])))
	}

	taskRoll := rng.Intn(3)
	task := prompt.TaskSort
	order := "asc"
	intentLabel := uint8(output.IntentSortAsc)
	var promptText string

	if taskRoll == 2 {
		task = prompt.TaskSum
		intentLabel = uint8(output.IntentSum)
		promptText = fitPrompt(rng, func() string { return buildSumPrompt(rng, values, lex, mode) })
	} else {
		if taskRoll == 1 {
			order = "desc"
			intentLabel = uint8(output.IntentSortDesc)
		}
		promptText = fitPrompt(rng, func() string { return buildSortPrompt(rng, order, values, lex, mode) })
	}

	targetText := prompt.BuildTarget(task, numbers, order)
	return output.Output{Prompt: promptText, Target: targetText, Intent: intentLabel}
}

func buildSortPrompt(rng *rand.Rand, order string, values []string, lex lexicon, mode genMode) string {
	opener := pick(rng, lex.sortOpeners)
	verb := pick(rng, lex.sortVerbs)
	noun := pick(rng, lex.listNouns)
	dir := pick(rng, lex.ascPhrases)
	if order == "desc" {
		dir = pick(rng, lex.descPhrases)
	}
	nums := formatNumbers(rng, values)

	patternMax := 8
	if mode == modeOOD {
		patternMax = 12
	}

	var text string
	switch rng.Intn(patternMax) {
	case 0:
		text = joinWords(opener, verb, "this", noun, nums, "in", dir)
	case 1:
		text = joinWords(opener, "please", verb, nums, dir)
	case 2:
		text = joinWords(opener, "take", nums, "and", verb, "them", dir)
	case 3:
		text = joinWords(opener, "i want", nums, verb, dir)
	case 4:
		text = joinWords(opener, "put", nums, "in", dir)
	case 5:
		text = joinWords(opener, verb, "the", noun, nums, "so it is", dir)
	case 6:
		text = joinWords(opener, "make", nums, "go", dir)
	case 7:
		text = joinWords(opener, "for", nums, "use", dir)
	case 8:
		text = joinWords("if you can", "arrange", nums, "in", dir)
	case 9:
		text = joinWords("transform", nums, "into", dir)
	case 10:
		text = joinWords("rewrite", nums, "with", dir)
	default:
		text = joinWords("process", "the", noun, nums, "under", dir)
	}

	text = maybeInjectNoise(rng, text)
	text = maybeCaseShift(rng, text)
	text = maybePunctuation(rng, text)
	return strings.TrimSpace(text)
}

func buildSumPrompt(rng *rand.Rand, values []string, lex lexicon, mode genMode) string {
	nums := formatNumbers(rng, values)
	verb := pick(rng, lex.sumVerbs)
	obj := pick(rng, lex.sumObjects)
	ask := pick(rng, lex.sumAsk)

	patternMax := 8
	if mode == modeOOD {
		patternMax = 12
	}

	var text string
	switch rng.Intn(patternMax) {
	case 0:
		text = joinWords("please", verb, obj, nums)
	case 1:
		text = joinWords(ask, obj, nums)
	case 2:
		text = joinWords("can you", verb, nums)
	case 3:
		text = joinWords("i need", "the total for", nums)
	case 4:
		text = joinWords("find", "the sum of", nums)
	case 5:
		text = joinWords("add up", nums)
	case 6:
		text = joinWords("sum", nums, "for me")
	case 7:
		text = joinWords("compute", "the total of", nums)
	case 8:
		text = joinWords("derive", "the aggregate over", nums)
	case 9:
		text = joinWords("combine", nums, "into one total")
	case 10:
		text = joinWords("resolve", nums, "to their total value")
	default:
		text = joinWords("what combined amount is", nums)
	}

	text = maybeInjectNoise(rng, text)
	text = maybeCaseShift(rng, text)
	text = maybePunctuation(rng, text)
	return strings.TrimSpace(text)
}

func formatNumbers(rng *rand.Rand, values []string) string {
	comma := strings.Join(values, ",")
	commaSpace := strings.Join(values, ", ")
	space := strings.Join(values, " ")
	if len(values) == 0 {
		return "[]"
	}
	andFmt := commaSpace
	if len(values) > 1 {
		andFmt = strings.Join(values[:len(values)-1], ", ") + " and " + values[len(values)-1]
	}

	switch rng.Intn(7) {
	case 0:
		return "[" + comma + "]"
	case 1:
		return "[" + commaSpace + "]"
	case 2:
		return "(" + commaSpace + ")"
	case 3:
		return space
	case 4:
		return andFmt
	case 5:
		return "numbers " + commaSpace
	default:
		return "values " + commaSpace
	}
}

func fitPrompt(rng *rand.Rand, f func() string) string {
	for range 12 {
		p := strings.TrimSpace(f())
		if len(p) <= output.MaxPromptBytes {
			return p
		}
	}
	p := strings.TrimSpace(f())
	if len(p) > output.MaxPromptBytes {
		return p[:output.MaxPromptBytes]
	}
	return p
}

func maybeInjectNoise(rng *rand.Rand, s string) string {
	if rng.Intn(100) < 20 {
		s = strings.ReplaceAll(s, "descending", "descendent")
	}
	if rng.Intn(100) < 20 {
		s = strings.ReplaceAll(s, "highest", "higest")
	}
	if rng.Intn(100) < 15 {
		s = strings.ReplaceAll(s, "ascending", "ascendent")
	}
	if rng.Intn(100) < 15 {
		s = strings.ReplaceAll(s, "sort", "sor")
	}
	if rng.Intn(100) < 10 {
		s = strings.ReplaceAll(s, "order", "ordeer")
	}
	return strings.Join(strings.Fields(s), " ")
}

func maybeCaseShift(rng *rand.Rand, s string) string {
	switch rng.Intn(5) {
	case 0:
		return strings.ToUpper(s)
	case 1:
		return simpleTitle(strings.ToLower(s))
	default:
		return s
	}
}

func simpleTitle(s string) string {
	var b strings.Builder
	b.Grow(len(s))

	startOfWord := true
	for _, r := range s {
		if unicode.IsSpace(r) {
			startOfWord = true
			b.WriteRune(r)
			continue
		}
		if startOfWord {
			b.WriteRune(unicode.ToTitle(r))
			startOfWord = false
			continue
		}
		b.WriteRune(r)
	}

	return b.String()
}

func maybePunctuation(rng *rand.Rand, s string) string {
	suffix := []string{"", ".", "?", "!", " please", " now", " asap"}
	return s + pick(rng, suffix)
}

func joinWords(parts ...string) string {
	filtered := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			filtered = append(filtered, p)
		}
	}
	return strings.Join(filtered, " ")
}

func pick(rng *rand.Rand, vals []string) string {
	return vals[rng.Intn(len(vals))]
}
