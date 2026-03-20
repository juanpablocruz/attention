//go:build !darwin || !arm64 || !cgo

package matrix

import "math"

func simdAddInPlace(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func simdSubScaled(dst, grad []float32, scale float32) {
	for i := range dst {
		dst[i] -= scale * grad[i]
	}
}

func simdReLU(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

func simdRowMax(row []float32) float32 {
	m := row[0]
	for i := 1; i < len(row); i++ {
		if row[i] > m {
			m = row[i]
		}
	}
	return m
}

func simdRowSumExpShift(row []float32, shift float32) float32 {
	s := float32(0)
	for i := range row {
		s += float32(math.Exp(float64(row[i] - shift)))
	}
	return s
}

func simdRowMean(row []float32) float32 {
	s := float32(0)
	for i := range row {
		s += row[i]
	}
	return s / float32(len(row))
}

func simdRowVariance(row []float32, mean float32) float32 {
	s := float32(0)
	for i := range row {
		d := row[i] - mean
		s += d * d
	}
	return s / float32(len(row))
}

func simdNormalizeRow(dst, src []float32, mean, invStd float32) {
	for i := range src {
		dst[i] = (src[i] - mean) * invStd
	}
}
