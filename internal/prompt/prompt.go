package prompt

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const (
	SortToken = 10
	ListToken = 11
	AscToken  = 12
	DescToken = 13
	SumToken  = 14
	PadToken  = 15

	MinListLen  = 3
	MaxListLen  = 7
	SequenceLen = 2 + MaxListLen + 1
	ListLen     = MaxListLen
	VocabSize   = 16
)

const (
	TaskSort = "sort"
	TaskSum  = "sum"
)

var punctuation = regexp.MustCompile(`[\[\],:?]`)
var numberPattern = regexp.MustCompile(`\d+`)
var bracketPattern = regexp.MustCompile(`\[(.*?)\]`)

type ParsedPrompt struct {
	Task    string
	Numbers [MaxListLen]uint8
	Count   int
	Order   string
}

type Tokenizer struct {
	vocab map[string]uint8
}

func NewTokenizer() *Tokenizer {
	return &Tokenizer{
		vocab: map[string]uint8{
			"sort": SortToken,
			"list": ListToken,
			"asc":  AscToken,
			"desc": DescToken,
			"sum":  SumToken,
			"pad":  PadToken,
		},
	}
}

var DefaultTokenizer = NewTokenizer()

func (t *Tokenizer) BuildPrompt(task string, numbers []uint8, order string) string {
	if len(numbers) == 0 {
		numbers = []uint8{0, 0, 0}
	}
	if len(numbers) > MaxListLen {
		numbers = numbers[:MaxListLen]
	}

	if task == TaskSum {
		parts := make([]string, 0, 3+len(numbers))
		parts = append(parts, "sum", "list", "[")
		for i := 0; i < len(numbers); i++ {
			parts = append(parts, strconv.Itoa(int(numbers[i])))
		}
		parts = append(parts, "]")
		return strings.Join(parts, " ")
	}

	parts := make([]string, 0, 4+len(numbers))
	parts = append(parts, "sort", "list", "[")
	for i := 0; i < len(numbers); i++ {
		parts = append(parts, strconv.Itoa(int(numbers[i])))
	}
	parts = append(parts, "]", strings.ToLower(order))
	return strings.Join(parts, " ")
}

func (t *Tokenizer) BuildTarget(task string, numbers []uint8, order string) string {
	if len(numbers) > MaxListLen {
		numbers = numbers[:MaxListLen]
	}

	if task == TaskSum {
		total := 0
		for i := 0; i < len(numbers); i++ {
			total += int(numbers[i])
		}
		out := make([]uint8, MaxListLen)
		out[MaxListLen-2] = uint8((total / 10) % 10)
		out[MaxListLen-1] = uint8(total % 10)
		return t.BuildPrompt(TaskSum, out, "")
	}

	sorted := make([]uint8, len(numbers))
	copy(sorted, numbers)
	slices.Sort(sorted)
	if strings.ToLower(order) == "desc" {
		slices.Reverse(sorted)
	}
	return t.BuildPrompt(TaskSort, sorted, order)
}

func (t *Tokenizer) Encode(text string) ([SequenceLen]uint8, error) {
	parsed, err := t.Parse(text)
	if err != nil {
		return [SequenceLen]uint8{}, err
	}
	return t.EncodeParsed(parsed)
}

func (t *Tokenizer) EncodeParsed(parsed ParsedPrompt) ([SequenceLen]uint8, error) {
	var tokens [SequenceLen]uint8
	for i := range SequenceLen {
		tokens[i] = t.vocab["pad"]
	}

	if parsed.Task == TaskSum {
		tokens[0] = t.vocab[TaskSum]
		tokens[1] = t.vocab["list"]
		for i := 0; i < parsed.Count && i < MaxListLen; i++ {
			tokens[i+2] = parsed.Numbers[i]
		}
		return tokens, nil
	}

	tokens[0] = t.vocab[TaskSort]
	tokens[1] = t.vocab["list"]
	for i := 0; i < parsed.Count && i < MaxListLen; i++ {
		tokens[i+2] = parsed.Numbers[i]
	}
	tokens[SequenceLen-1] = t.vocab[parsed.Order]

	return tokens, nil
}

func (t *Tokenizer) Parse(text string) (ParsedPrompt, error) {
	var parsed ParsedPrompt
	clean := strings.ToLower(strings.TrimSpace(text))
	if clean == "" {
		return parsed, fmt.Errorf("empty prompt")
	}

	task, err := parseTask(clean)
	if err != nil {
		return parsed, err
	}

	numbers, numCount, err := parseNumbers(clean)
	if err != nil {
		return parsed, err
	}

	parsed.Task = task
	parsed.Numbers = numbers
	parsed.Count = numCount
	if parsed.Count < MinListLen || parsed.Count > MaxListLen {
		return parsed, fmt.Errorf("expected list length between %d and %d, got %d", MinListLen, MaxListLen, parsed.Count)
	}
	if task == TaskSort {
		order, err := parseOrder(clean)
		if err != nil {
			return parsed, err
		}
		parsed.Order = order
	}
	return parsed, nil
}

func parseTask(clean string) (string, error) {
	fields := strings.Fields(punctuation.ReplaceAllString(clean, " "))
	task := ""
	for _, f := range fields {
		switch f {
		case "sum", "add", "total":
			if task == TaskSort {
				return "", fmt.Errorf("prompt contains both sort and sum cues")
			}
			task = TaskSum
		case "sort", "order":
			if task == TaskSum {
				return "", fmt.Errorf("prompt contains both sort and sum cues")
			}
			task = TaskSort
		}
	}
	if task == "" {
		return "", fmt.Errorf("could not infer task (sort/order or sum/add)")
	}
	return task, nil
}

func parseOrder(clean string) (string, error) {
	fields := strings.Fields(punctuation.ReplaceAllString(clean, " "))
	joined := strings.Join(fields, " ")

	if strings.Contains(joined, "smallest to largest") || strings.Contains(joined, "lowest to highest") {
		return "asc", nil
	}
	if strings.Contains(joined, "largest to smallest") || strings.Contains(joined, "highest to lowest") {
		return "desc", nil
	}

	order := ""
	for _, f := range fields {
		switch f {
		case "asc", "ascending", "increasing":
			if order == "desc" {
				return "", fmt.Errorf("prompt contains both ascending and descending cues")
			}
			order = "asc"
		case "desc", "descending", "decreasing", "reverse":
			if order == "asc" {
				return "", fmt.Errorf("prompt contains both ascending and descending cues")
			}
			order = "desc"
		}
	}

	if order == "" {
		return "", fmt.Errorf("could not infer order (use asc/ascending or desc/descending)")
	}

	return order, nil
}

func parseNumbers(clean string) ([MaxListLen]uint8, int, error) {
	var numbers [MaxListLen]uint8

	raw := ""
	if m := bracketPattern.FindStringSubmatch(clean); len(m) == 2 {
		raw = m[1]
	} else {
		raw = clean
	}

	matches := numberPattern.FindAllString(raw, -1)
	if len(matches) < MinListLen || len(matches) > MaxListLen {
		return numbers, 0, fmt.Errorf("expected between %d and %d numbers, got %d", MinListLen, MaxListLen, len(matches))
	}

	for i, s := range matches {
		n, err := strconv.Atoi(s)
		if err != nil {
			return numbers, 0, fmt.Errorf("invalid number %q", s)
		}
		if n < 0 || n > 9 {
			return numbers, 0, fmt.Errorf("number out of range: %d", n)
		}
		numbers[i] = uint8(n)
	}

	return numbers, len(matches), nil
}

func BuildPrompt(task string, numbers []uint8, order string) string {
	return DefaultTokenizer.BuildPrompt(task, numbers, order)
}

func BuildTarget(task string, numbers []uint8, order string) string {
	return DefaultTokenizer.BuildTarget(task, numbers, order)
}

func Encode(text string) ([SequenceLen]uint8, error) {
	return DefaultTokenizer.Encode(text)
}

func Parse(text string) (ParsedPrompt, error) {
	return DefaultTokenizer.Parse(text)
}
