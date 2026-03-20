package intent

import (
	"encoding/gob"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLabelToTaskOrder(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		label     Label
		wantTask  string
		wantOrder string
		wantErr   bool
	}{
		{name: "sort asc", label: SortAsc, wantTask: "sort", wantOrder: "asc"},
		{name: "sort desc", label: SortDesc, wantTask: "sort", wantOrder: "desc"},
		{name: "sum", label: Sum, wantTask: "sum"},
		{name: "invalid", label: Label(99), wantErr: true},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			task, order, err := LabelToTaskOrder(tt.label)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("LabelToTaskOrder(%d) error = nil, want error", tt.label)
				}
				return
			}
			if err != nil {
				t.Fatalf("LabelToTaskOrder(%d) error = %v", tt.label, err)
			}
			if task != tt.wantTask || order != tt.wantOrder {
				t.Fatalf("LabelToTaskOrder(%d) = (%q, %q), want (%q, %q)", tt.label, task, order, tt.wantTask, tt.wantOrder)
			}
		})
	}
}

func TestTokenizeAndFeaturize(t *testing.T) {
	t.Parallel()

	tokens := tokenize("Sort LIST [9, 1, 0] Asc")
	wantTokens := []string{"sort", "list", "9", "1", "0", "asc"}
	if !reflect.DeepEqual(tokens, wantTokens) {
		t.Fatalf("tokenize() = %v, want %v", tokens, wantTokens)
	}

	featuresA := featurize("sum list [1,2,3]")
	featuresB := featurize("sum list [1,2,3]")
	if !reflect.DeepEqual(featuresA, featuresB) {
		t.Fatal("featurize() is not deterministic for identical input")
	}
	if len(featuresA) == 0 {
		t.Fatal("featurize() returned no features")
	}
	if gotA, gotB := hashFeature("sum_list"), hashFeature("sum_list"); gotA != gotB {
		t.Fatal("hashFeature() is not deterministic")
	}
}

func TestTrainStepImprovesTargetProbability(t *testing.T) {
	t.Parallel()

	model := NewModel()
	text := "sort list [ 1 2 3 ] asc"

	_, before := model.Predict(text)
	for range 200 {
		model.TrainStep(text, SortAsc, 0.05)
	}
	label, after := model.Predict(text)

	if label != SortAsc {
		t.Fatalf("Predict() label = %v, want %v", label, SortAsc)
	}
	if after[int(SortAsc)] <= before[int(SortAsc)] {
		t.Fatalf("target probability did not improve: before=%v after=%v", before[int(SortAsc)], after[int(SortAsc)])
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "intent.gob")

	model := NewModel()
	model.B2[SortAsc] = 1.25
	model.W2[SortAsc][0] = 2.5

	if err := model.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	got, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if !reflect.DeepEqual(got, model) {
		t.Fatalf("Load() = %#v, want %#v", got, model)
	}
}

func TestLoadRejectsInvalidShape(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "invalid.gob")

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("os.Create() error = %v", err)
	}
	invalid := Model{
		W1: make([][]float32, 1),
		B1: make([]float32, 1),
		W2: make([][]float32, 1),
		B2: make([]float32, 1),
	}
	if err := gob.NewEncoder(f).Encode(invalid); err != nil {
		_ = f.Close()
		t.Fatalf("Encode() error = %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	if _, err := Load(path); err == nil {
		t.Fatal("Load() error = nil, want error")
	}
}
