//go:build !darwin || !arm64 || !cgo

package matrix

// Fallback when true ARM NEON intrinsics backend is unavailable.
func mulNeonInto(dst, a, b *Matrix) *Matrix {
	return mulGenericInto(dst, a, b)
}

func mulNeonPackedInto(dst, a, b *Matrix) *Matrix {
	return mulGenericInto(dst, a, b)
}

func mulNeonPackedIntoMatrix(dst, a *Matrix, b *PackedMatrix) *Matrix {
	if b == nil {
		return nil
	}
	return mulGenericInto(dst, a, b.Original)
}
