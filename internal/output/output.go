package output

type Output struct {
	Source [5]uint8
	Target [5]uint8
}

func DecodeRecord(buf []byte) Output {
	var out Output
	copy(out.Source[:], buf[0:5])
	copy(out.Target[:], buf[5:10])
	return out
}
