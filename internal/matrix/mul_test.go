package matrix

import (
	"math"
	"testing"
)

func filledMatrix(rows, cols int) *Matrix {
	m := NewZeroMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Vec[i][j] = float32(((i+1)*(j+3))%17-8) / 8
		}
	}
	return m
}

func TestMulIntoMatchesMul(t *testing.T) {
	tests := []struct {
		name string
		m    int
		n    int
		k    int
	}{
		{name: "tiny", m: 4, n: 4, k: 4},
		{name: "medium", m: 17, n: 19, k: 13},
		{name: "tail-heavy", m: 7, n: 10, k: 9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := filledMatrix(tt.m, tt.k)
			b := filledMatrix(tt.k, tt.n)

			want := a.Mul(b)
			if want == nil {
				t.Fatal("Mul returned nil")
			}

			dst := NewZeroMatrix(tt.m, tt.n)
			got := a.MulInto(dst, b)
			if got != dst {
				t.Fatal("MulInto did not reuse the provided destination")
			}
			if got.Rows != tt.m || got.Cols != tt.n {
				t.Fatalf("MulInto returned wrong shape: got %dx%d want %dx%d", got.Rows, got.Cols, tt.m, tt.n)
			}

			for i := range want.Data {
				diff := math.Abs(float64(got.Data[i] - want.Data[i]))
				if diff > 1e-5 {
					t.Fatalf("result mismatch at %d: got %v want %v", i, got.Data[i], want.Data[i])
				}
			}
		})
	}
}

func TestMulNeonPackedIntoMatchesMul(t *testing.T) {
	tests := []struct {
		name string
		m    int
		n    int
		k    int
	}{
		{name: "small-square", m: 8, n: 8, k: 8},
		{name: "tail-columns", m: 13, n: 10, k: 11},
		{name: "larger-square", m: 32, n: 32, k: 32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := filledMatrix(tt.m, tt.k)
			b := filledMatrix(tt.k, tt.n)

			want := a.Mul(b)
			if want == nil {
				t.Fatal("Mul returned nil")
			}

			dst := NewZeroMatrix(tt.m, tt.n)
			got := mulNeonPackedInto(dst, a, b)
			if got != dst {
				t.Fatal("mulNeonPackedInto did not reuse the provided destination")
			}

			for i := range want.Data {
				diff := math.Abs(float64(got.Data[i] - want.Data[i]))
				if diff > 1e-5 {
					t.Fatalf("result mismatch at %d: got %v want %v", i, got.Data[i], want.Data[i])
				}
			}
		})
	}
}

func TestMulPackedIntoMatchesMul(t *testing.T) {
	tests := []struct {
		name string
		m    int
		n    int
		k    int
	}{
		{name: "medium", m: 17, n: 19, k: 13},
		{name: "larger", m: 32, n: 32, k: 32},
		{name: "tail-columns", m: 15, n: 10, k: 12},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := filledMatrix(tt.m, tt.k)
			b := filledMatrix(tt.k, tt.n)
			packed := b.PackB()

			want := a.Mul(b)
			if want == nil {
				t.Fatal("Mul returned nil")
			}

			dst := NewZeroMatrix(tt.m, tt.n)
			got := a.MulPackedInto(dst, packed)
			if got != dst {
				t.Fatal("MulPackedInto did not reuse the provided destination")
			}

			for i := range want.Data {
				diff := math.Abs(float64(got.Data[i] - want.Data[i]))
				if diff > 1e-5 {
					t.Fatalf("result mismatch at %d: got %v want %v", i, got.Data[i], want.Data[i])
				}
			}
		})
	}
}

func BenchmarkMul(b *testing.B) {
	benchmarks := []struct {
		name string
		m    int
		n    int
		k    int
	}{
		{name: "4x4x4", m: 4, n: 4, k: 4},
		{name: "8x8x8", m: 8, n: 8, k: 8},
		{name: "16x16x16", m: 16, n: 16, k: 16},
		{name: "20x20x20", m: 20, n: 20, k: 20},
		{name: "24x24x24", m: 24, n: 24, k: 24},
		{name: "16x32x32", m: 16, n: 32, k: 32},
		{name: "24x32x32", m: 24, n: 32, k: 32},
		{name: "32x32x32", m: 32, n: 32, k: 32},
		{name: "48x48x48", m: 48, n: 48, k: 48},
		{name: "64x64x64", m: 64, n: 64, k: 64},
		{name: "32x64x64", m: 32, n: 64, k: 64},
		{name: "64x64x128", m: 64, n: 64, k: 128},
	}

	for _, bm := range benchmarks {
		a := filledMatrix(bm.m, bm.k)
		cases := []struct {
			name string
			run  func(*testing.B, *Matrix)
		}{
			{
				name: "mul",
				run: func(b *testing.B, other *Matrix) {
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_ = a.Mul(other)
					}
				},
			},
			{
				name: "mul-into",
				run: func(b *testing.B, other *Matrix) {
					dst := NewZeroMatrix(bm.m, bm.n)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_ = a.MulInto(dst, other)
					}
				},
			},
			{
				name: "generic-into",
				run: func(b *testing.B, other *Matrix) {
					dst := NewZeroMatrix(bm.m, bm.n)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_ = mulGenericInto(dst, a, other)
					}
				},
			},
			{
				name: "packed-into",
				run: func(b *testing.B, other *Matrix) {
					dst := NewZeroMatrix(bm.m, bm.n)
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_ = mulNeonPackedInto(dst, a, other)
					}
				},
			},
			{
				name: "prepacked-into",
				run: func(b *testing.B, other *Matrix) {
					dst := NewZeroMatrix(bm.m, bm.n)
					packed := other.PackB()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						_ = a.MulPackedInto(dst, packed)
					}
				},
			},
		}

		for _, tc := range cases {
			tc := tc
			b.Run(bm.name+"/"+tc.name, func(b *testing.B) {
				other := filledMatrix(bm.k, bm.n)
				b.ReportAllocs()
				tc.run(b, other)
			})
		}
	}
}

func BenchmarkPackB(b *testing.B) {
	benchmarks := []struct {
		name string
		rows int
		cols int
	}{
		{name: "32x32", rows: 32, cols: 32},
		{name: "64x64", rows: 64, cols: 64},
		{name: "64x128", rows: 64, cols: 128},
	}

	for _, bm := range benchmarks {
		m := filledMatrix(bm.rows, bm.cols)
		b.Run(bm.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = m.PackB()
			}
		})
	}
}
