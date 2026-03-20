package trainer

import (
	"fmt"
	"os"
	"runtime"

	"github.com/juanpablocruz/attention/gen/internal/intent"
	"github.com/juanpablocruz/attention/gen/internal/output"
	"github.com/juanpablocruz/attention/gen/pkg/decode"
)

func openDataset(path string) (*os.File, int64, int, error) {
	stat, err := os.Stat(path)
	if err != nil {
		return nil, 0, 0, err
	}

	size := stat.Size()
	if size%decode.RecordSize != 0 {
		return nil, 0, 0, fmt.Errorf("invalid file size: %d is not divisible by %d", size, decode.RecordSize)
	}

	totalRecords := size / decode.RecordSize
	if totalRecords == 0 {
		return nil, 0, 0, fmt.Errorf("empty dataset")
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}

	workers := max(min(runtime.NumCPU(), int(totalRecords)), 1)
	return f, totalRecords, workers, nil
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

func isIntentValidationSample(promptText string) bool {
	var h uint32 = 2166136261
	for i := 0; i < len(promptText); i++ {
		h ^= uint32(promptText[i])
		h *= 16777619
	}
	return h%10 == 0
}

