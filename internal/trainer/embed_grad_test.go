package trainer

import (
	"testing"

	"github.com/juanpablocruz/attention/gen/internal/attention"
	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/internal/prompt"
)

func benchmarkSample() ([]uint8, []uint8, string, int) {
	numbers := []uint8{3, 1, 4, 2}
	source, err := prompt.EncodeStructured(prompt.TaskSort, "asc", numbers)
	if err != nil {
		panic(err)
	}
	target, err := prompt.EncodeStructured(prompt.TaskSort, "asc", []uint8{1, 2, 3, 4})
	if err != nil {
		panic(err)
	}
	return source[:], target[:], prompt.TaskSort, len(numbers)
}

func BenchmarkComputeSampleGrad(b *testing.B) {
	source, target, task, count := benchmarkSample()

	b.Run("unpacked", func(b *testing.B) {
		model := attention.NewModel(prompt.VocabSize, defaultModelDim, defaultNumHeads)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := computeSampleGrad(source, target, task, count, model, nil); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("prepacked-model", func(b *testing.B) {
		model := attention.NewModel(prompt.VocabSize, defaultModelDim, defaultNumHeads)
		packed := model.Pack()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := computeSampleGrad(source, target, task, count, model, packed); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkPrepareBatch(b *testing.B) {
	batch := make([]output.Output, 256)
	for i := range batch {
		batch[i] = output.Output{
			Prompt: "sort asc 3 1 4 2",
			Target: "1 2 3 4",
			Intent: output.IntentSortAsc,
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := prepareBatch(batch); err != nil {
			b.Fatal(err)
		}
	}
}
