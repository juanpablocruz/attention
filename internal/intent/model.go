package intent

import (
	"encoding/gob"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"regexp"
	"strings"
	"time"
)

type Label int

const (
	SortAsc Label = iota
	SortDesc
	Sum
	NumLabels
)

const (
	FeatureDim = 8192
	HiddenDim  = 128
)

type Model struct {
	W1 [][]float32
	B1 []float32
	W2 [][]float32
	B2 []float32
}

// tokenRe extracts lowercase alphabetic words and digit runs as the basic intent tokens.
var tokenRe = regexp.MustCompile(`[a-z]+|\d+`)

func NewModel() *Model {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	m := &Model{
		W1: make([][]float32, HiddenDim),
		B1: make([]float32, HiddenDim),
		W2: make([][]float32, NumLabels),
		B2: make([]float32, NumLabels),
	}
	for h := range HiddenDim {
		m.W1[h] = make([]float32, FeatureDim)
		for i := range FeatureDim {
			m.W1[h][i] = (rng.Float32()*2 - 1) * 0.02
		}
	}
	for c := range int(NumLabels) {
		m.W2[c] = make([]float32, HiddenDim)
		for h := range HiddenDim {
			m.W2[c][h] = (rng.Float32()*2 - 1) * 0.02
		}
	}
	return m
}

func LabelToTaskOrder(l Label) (task, order string, err error) {
	switch l {
	case SortAsc:
		return "sort", "asc", nil
	case SortDesc:
		return "sort", "desc", nil
	case Sum:
		return "sum", "", nil
	default:
		return "", "", fmt.Errorf("unknown label: %d", l)
	}
}

func (m *Model) TrainStep(text string, label Label, lr float32) {
	x := featurize(text)
	hiddenPre, hiddenAct, probs := m.forward(x)

	dLogits := make([]float32, NumLabels)
	for c := range int(NumLabels) {
		dLogits[c] = probs[c]
	}
	dLogits[int(label)] -= 1

	dHidden := make([]float32, HiddenDim)
	for h := range HiddenDim {
		g := float32(0)
		for c := range int(NumLabels) {
			g += dLogits[c] * m.W2[c][h]
		}
		if hiddenPre[h] <= 0 {
			g = 0
		}
		dHidden[h] = g
	}

	for c := range int(NumLabels) {
		m.B2[c] -= lr * dLogits[c]
		for h := range HiddenDim {
			m.W2[c][h] -= lr * dLogits[c] * hiddenAct[h]
		}
	}

	for h := range HiddenDim {
		m.B1[h] -= lr * dHidden[h]
		for idx, v := range x {
			m.W1[h][idx] -= lr * dHidden[h] * v
		}
	}
}

func (m *Model) Predict(text string) (Label, []float32) {
	x := featurize(text)
	_, _, probs := m.forward(x)
	best := 0
	for i := 1; i < len(probs); i++ {
		if probs[i] > probs[best] {
			best = i
		}
	}
	return Label(best), probs
}

func (m *Model) Save(path string) (err error) {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() {
		err = errors.Join(err, f.Close())
	}()
	return gob.NewEncoder(f).Encode(m)
}

func Load(path string) (_ *Model, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		err = errors.Join(err, f.Close())
	}()
	var m Model
	if err := gob.NewDecoder(f).Decode(&m); err != nil {
		return nil, err
	}
	if len(m.W1) != HiddenDim || len(m.B1) != HiddenDim || len(m.W2) != int(NumLabels) || len(m.B2) != int(NumLabels) {
		return nil, fmt.Errorf("invalid intent model shape")
	}
	for h := range HiddenDim {
		if len(m.W1[h]) != FeatureDim {
			return nil, fmt.Errorf("invalid intent model shape")
		}
	}
	for c := range int(NumLabels) {
		if len(m.W2[c]) != HiddenDim {
			return nil, fmt.Errorf("invalid intent model shape")
		}
	}
	return &m, nil
}

func featurize(text string) map[int]float32 {
	toks := tokenize(text)
	feats := make(map[int]float32, len(toks)*8)
	for i := range toks {
		// Hash each word unigram into the fixed-size feature space.
		feats[hashFeature("w:"+toks[i])] += 1
		if i+1 < len(toks) {
			// Hash adjacent word bigrams into the same feature space.
			feats[hashFeature("b:"+toks[i]+"_"+toks[i+1])] += 1
		}
	}

	norm := normalizeForCharNGrams(text)
	for n := 3; n <= 5; n++ {
		if len(norm) < n {
			continue
		}
		for i := 0; i <= len(norm)-n; i++ {
			ng := norm[i : i+n]
			feats[hashFeature("c:"+ng)] += 1
		}
	}
	return feats
}

func normalizeForCharNGrams(text string) string {
	t := strings.ToLower(text)
	var b strings.Builder
	b.Grow(len(t) + 2)
	b.WriteByte('^')
	for i := 0; i < len(t); i++ {
		c := t[i]
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == ' ' {
			b.WriteByte(c)
		} else {
			b.WriteByte(' ')
		}
	}
	b.WriteByte('$')
	return b.String()
}

func tokenize(text string) []string {
	t := strings.ToLower(text)
	// Tokenization is regex-based.
	return tokenRe.FindAllString(t, -1)
}

func hashFeature(s string) int {
	h := fnv.New32a()
	_, _ = h.Write([]byte(s))
	return int(h.Sum32() % FeatureDim)
}

func (m *Model) forward(x map[int]float32) ([]float32, []float32, []float32) {
	hiddenPre := make([]float32, HiddenDim)
	hiddenAct := make([]float32, HiddenDim)
	for h := range HiddenDim {
		s := m.B1[h]
		for idx, v := range x {
			s += m.W1[h][idx] * v
		}
		hiddenPre[h] = s
		// ReLU activation.
		if s > 0 {
			hiddenAct[h] = s
		}
	}

	logits := make([]float32, NumLabels)
	for c := range int(NumLabels) {
		s := m.B2[c]
		for h := range HiddenDim {
			s += m.W2[c][h] * hiddenAct[h]
		}
		logits[c] = s
	}
	return hiddenPre, hiddenAct, softmax(logits)
}

func softmax(logits []float32) []float32 {
	maxV := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxV {
			maxV = logits[i]
		}
	}
	exps := make([]float32, len(logits))
	sum := float32(0)
	for i := range logits {
		e := float32(math.Exp(float64(logits[i] - maxV)))
		exps[i] = e
		sum += e
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}
