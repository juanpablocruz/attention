package prompt

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

const (
	PadToken  = 14
	SortToken = 10
	ListToken = 11
	AscToken  = 12
	DescToken = 13

	SequenceLen = 8
	ListLen     = 5
	VocabSize   = 15
)

var punctuation = regexp.MustCompile(`[\[\],]`)
var numberPattern = regexp.MustCompile(`\d+`)
var bracketPattern = regexp.MustCompile(`\[(.*?)\]`)

type ParsedPrompt struct {
	Numbers [ListLen]uint8
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
			"pad":  PadToken,
		},
	}
}

var DefaultTokenizer = NewTokenizer()

func (t *Tokenizer) BuildPrompt(numbers [ListLen]uint8, order string) string {
	parts := make([]string, 0, 4+ListLen)
	parts = append(parts, "sort", "list", "[")
	for i := 0; i < ListLen; i++ {
		parts = append(parts, strconv.Itoa(int(numbers[i])))
	}
	parts = append(parts, "]", strings.ToLower(order))
	return strings.Join(parts, " ")
}

func (t *Tokenizer) BuildTarget(numbers [ListLen]uint8, order string) string {
	sorted := numbers
	slices.Sort(sorted[:])
	if strings.ToLower(order) == "desc" {
		slices.Reverse(sorted[:])
	}
	return t.BuildPrompt(sorted, order)
}

func (t *Tokenizer) Encode(text string) ([SequenceLen]uint8, error) {
	var tokens [SequenceLen]uint8
	parsed, err := t.Parse(text)
	if err != nil {
		return tokens, err
	}

	tokens[0] = t.vocab["sort"]
	tokens[1] = t.vocab["list"]
	for i := 0; i < ListLen; i++ {
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

	order, err := parseOrder(clean)
	if err != nil {
		return parsed, err
	}

	numbers, err := parseNumbers(clean)
	if err != nil {
		return parsed, err
	}

	parsed.Order = order
	parsed.Numbers = numbers
	return parsed, nil
}

func parseOrder(clean string) (string, error) {
	clean = punctuation.ReplaceAllString(clean, " ")
	fields := strings.Fields(clean)

	order := ""
	for _, f := range fields {
		switch f {
		case "asc", "ascending", "increasing", "smallest", "lowest":
			if order == "desc" {
				return "", fmt.Errorf("prompt contains both ascending and descending cues")
			}
			order = "asc"
		case "desc", "descending", "decreasing", "largest", "highest":
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

func parseNumbers(clean string) ([ListLen]uint8, error) {
	var numbers [ListLen]uint8

	raw := ""
	if m := bracketPattern.FindStringSubmatch(clean); len(m) == 2 {
		raw = m[1]
	} else {
		raw = clean
	}

	matches := numberPattern.FindAllString(raw, -1)
	if len(matches) != ListLen {
		return numbers, fmt.Errorf("expected %d numbers, got %d", ListLen, len(matches))
	}

	for i, s := range matches {
		n, err := strconv.Atoi(s)
		if err != nil {
			return numbers, fmt.Errorf("invalid number %q", s)
		}
		if n < 0 || n > 9 {
			return numbers, fmt.Errorf("number out of range: %d", n)
		}
		numbers[i] = uint8(n)
	}

	return numbers, nil
}

func BuildPrompt(numbers [ListLen]uint8, order string) string {
	return DefaultTokenizer.BuildPrompt(numbers, order)
}

func BuildTarget(numbers [ListLen]uint8, order string) string {
	return DefaultTokenizer.BuildTarget(numbers, order)
}

func Encode(text string) ([SequenceLen]uint8, error) {
	return DefaultTokenizer.Encode(text)
}

func Parse(text string) ([ListLen]uint8, string, error) {
	parsed, err := DefaultTokenizer.Parse(text)
	if err != nil {
		return [ListLen]uint8{}, "", err
	}
	return parsed.Numbers, parsed.Order, nil
}
