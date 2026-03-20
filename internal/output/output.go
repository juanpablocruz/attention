package output

const (
	MaxPromptBytes = 160
	MaxTargetBytes = 64
	RecordSize     = 3 + MaxPromptBytes + MaxTargetBytes

	IntentSortAsc  = 0
	IntentSortDesc = 1
	IntentSum      = 2
)

type Output struct {
	Prompt string
	Target string
	Intent uint8
}

func (o Output) EncodeRecord() [RecordSize]byte {
	var buf [RecordSize]byte
	prompt := []byte(o.Prompt)
	if len(prompt) > MaxPromptBytes {
		prompt = prompt[:MaxPromptBytes]
	}
	target := []byte(o.Target)
	if len(target) > MaxTargetBytes {
		target = target[:MaxTargetBytes]
	}

	buf[0] = byte(len(prompt))
	copy(buf[1:1+MaxPromptBytes], prompt)

	buf[1+MaxPromptBytes] = byte(len(target))
	copy(buf[2+MaxPromptBytes:], target)
	buf[2+MaxPromptBytes+MaxTargetBytes] = o.Intent

	return buf
}

func DecodeRecord(buf []byte) Output {
	promptLen := min(int(buf[0]), MaxPromptBytes)

	targetLen := min(int(buf[1+MaxPromptBytes]), MaxTargetBytes)

	prompt := string(buf[1 : 1+promptLen])
	target := string(buf[2+MaxPromptBytes : 2+MaxPromptBytes+targetLen])
	intent := buf[2+MaxPromptBytes+MaxTargetBytes]

	return Output{Prompt: prompt, Target: target, Intent: intent}
}
