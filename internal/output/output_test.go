package output

import (
	"strings"
	"testing"
)

func TestEncodeDecodeRecordRoundTrip(t *testing.T) {
	t.Parallel()

	want := Output{
		Prompt: "sort list [ 1 2 3 ] asc",
		Target: "sort list [ 1 2 3 ] asc",
		Intent: IntentSortAsc,
	}

	record := want.EncodeRecord()
	got := DecodeRecord(record[:])
	if got != want {
		t.Fatalf("DecodeRecord(EncodeRecord()) = %#v, want %#v", got, want)
	}
}

func TestEncodeRecordTruncatesFields(t *testing.T) {
	t.Parallel()

	in := Output{
		Prompt: strings.Repeat("p", MaxPromptBytes+10),
		Target: strings.Repeat("t", MaxTargetBytes+7),
		Intent: IntentSum,
	}

	record := in.EncodeRecord()
	got := DecodeRecord(record[:])
	if len(got.Prompt) != MaxPromptBytes {
		t.Fatalf("prompt length = %d, want %d", len(got.Prompt), MaxPromptBytes)
	}
	if len(got.Target) != MaxTargetBytes {
		t.Fatalf("target length = %d, want %d", len(got.Target), MaxTargetBytes)
	}
	if got.Intent != in.Intent {
		t.Fatalf("intent = %d, want %d", got.Intent, in.Intent)
	}
}
