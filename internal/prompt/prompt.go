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
	VocabSize   = 16
)

const (
	TaskSort = "sort"
	TaskSum  = "sum"
)

var numberPattern = regexp.MustCompile(`\d+`)
var bracketPattern = regexp.MustCompile(`\[(.*?)\]`)

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

func (t *Tokenizer) EncodeStructured(task, order string, numbers []uint8) ([SequenceLen]uint8, error) {
	if task != TaskSort && task != TaskSum {
		return [SequenceLen]uint8{}, fmt.Errorf("invalid task: %s", task)
	}
	if task == TaskSort && order != "asc" && order != "desc" {
		return [SequenceLen]uint8{}, fmt.Errorf("invalid sort order: %s", order)
	}
	if len(numbers) < MinListLen || len(numbers) > MaxListLen {
		return [SequenceLen]uint8{}, fmt.Errorf("expected between %d and %d numbers, got %d", MinListLen, MaxListLen, len(numbers))
	}

	var tokens [SequenceLen]uint8
	for i := range SequenceLen {
		tokens[i] = t.vocab["pad"]
	}

	if task == TaskSum {
		tokens[0] = t.vocab[TaskSum]
		tokens[1] = t.vocab["list"]
		for i := range numbers {
			if numbers[i] > 9 {
				return [SequenceLen]uint8{}, fmt.Errorf("number out of range: %d", numbers[i])
			}
			tokens[i+2] = numbers[i]
		}
		return tokens, nil
	}

	tokens[0] = t.vocab[TaskSort]
	tokens[1] = t.vocab["list"]
	for i := range numbers {
		if numbers[i] > 9 {
			return [SequenceLen]uint8{}, fmt.Errorf("number out of range: %d", numbers[i])
		}
		tokens[i+2] = numbers[i]
	}
	tokens[SequenceLen-1] = t.vocab[order]

	return tokens, nil
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

func ExtractNumbers(text string) ([]uint8, error) {
	clean := strings.ToLower(strings.TrimSpace(text))
	numbers, count, err := parseNumbers(clean)
	if err != nil {
		return nil, err
	}
	out := make([]uint8, count)
	copy(out, numbers[:count])
	return out, nil
}

func BuildPrompt(task string, numbers []uint8, order string) string {
	return DefaultTokenizer.BuildPrompt(task, numbers, order)
}

func BuildTarget(task string, numbers []uint8, order string) string {
	return DefaultTokenizer.BuildTarget(task, numbers, order)
}

func EncodeStructured(task, order string, numbers []uint8) ([SequenceLen]uint8, error) {
	return DefaultTokenizer.EncodeStructured(task, order, numbers)
}
