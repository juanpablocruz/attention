package output

const (
	MaxPromptBytes = 64
	MaxTargetBytes = 64
	RecordSize     = 2 + MaxPromptBytes + MaxTargetBytes
)

type Output struct {
	Prompt string
	Target string
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

	return buf
}

func DecodeRecord(buf []byte) Output {
	promptLen := min(int(buf[0]), MaxPromptBytes)

	targetLen := min(int(buf[1+MaxPromptBytes]), MaxTargetBytes)

	prompt := string(buf[1 : 1+promptLen])
	target := string(buf[2+MaxPromptBytes : 2+MaxPromptBytes+targetLen])

	return Output{Prompt: prompt, Target: target}
}
