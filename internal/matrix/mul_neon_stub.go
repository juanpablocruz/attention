//go:build !darwin || !arm64 || !cgo

package matrix

// Fallback when true ARM NEON intrinsics backend is unavailable.
func mulNeon(a, b *Matrix) *Matrix {
	return mulGeneric(a, b)
}
