package prompt

import (
	"reflect"
	"testing"
)

func TestBuildTarget(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		task   string
		order  string
		input  []uint8
		target string
	}{
		{
			name:   "sort asc",
			task:   TaskSort,
			order:  "asc",
			input:  []uint8{5, 2, 4},
			target: "sort list [ 2 4 5 ] asc",
		},
		{
			name:   "sort desc",
			task:   TaskSort,
			order:  "desc",
			input:  []uint8{5, 2, 4},
			target: "sort list [ 5 4 2 ] desc",
		},
		{
			name:   "sum",
			task:   TaskSum,
			input:  []uint8{7, 8, 9},
			target: "sum list [ 0 0 0 0 0 2 4 ]",
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := BuildTarget(tt.task, tt.input, tt.order); got != tt.target {
				t.Fatalf("BuildTarget(%q, %v, %q) = %q, want %q", tt.task, tt.input, tt.order, got, tt.target)
			}
		})
	}
}

func TestEncodeStructured(t *testing.T) {
	t.Parallel()

	got, err := EncodeStructured(TaskSort, "desc", []uint8{3, 1, 2})
	if err != nil {
		t.Fatalf("EncodeStructured error = %v", err)
	}

	want := [SequenceLen]uint8{SortToken, ListToken, 3, 1, 2, PadToken, PadToken, PadToken, PadToken, DescToken}
	if got != want {
		t.Fatalf("EncodeStructured() = %v, want %v", got, want)
	}
}

func TestExtractNumbers(t *testing.T) {
	t.Parallel()

	got, err := ExtractNumbers("sum list [9, 1, 0]")
	if err != nil {
		t.Fatalf("ExtractNumbers error = %v", err)
	}

	want := []uint8{9, 1, 0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ExtractNumbers() = %v, want %v", got, want)
	}
}
