//go:build darwin && arm64 && cgo

package matrix

import (
	"sync"
	"unsafe"
)

/*
#cgo CFLAGS: -O3 -ffast-math
#include <arm_neon.h>

// A: MxK row-major
// B: KxN row-major
// C: MxN row-major
// Tiled 4x4 micro-kernel. We vectorize across 4 output columns at once.
static void neon_sgemm_rowmajor_tiled(
    const float* __restrict A,
    const float* __restrict B,
    float* __restrict C,
    int M, int N, int K) {
    int i, j, k;

    for (i = 0; i + 4 <= M; i += 4) {
        for (j = 0; j + 4 <= N; j += 4) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            const float* a0 = A + (size_t)(i + 0) * K;
            const float* a1 = A + (size_t)(i + 1) * K;
            const float* a2 = A + (size_t)(i + 2) * K;
            const float* a3 = A + (size_t)(i + 3) * K;
            const float* pa0 = a0;
            const float* pa1 = a1;
            const float* pa2 = a2;
            const float* pa3 = a3;
            const float* pb = B + j;

            for (k = 0; k < K; k++, pb += N) {
                float32x4_t bvec = vld1q_f32(pb);
                acc0 = vfmaq_n_f32(acc0, bvec, *pa0++);
                acc1 = vfmaq_n_f32(acc1, bvec, *pa1++);
                acc2 = vfmaq_n_f32(acc2, bvec, *pa2++);
                acc3 = vfmaq_n_f32(acc3, bvec, *pa3++);
            }

            vst1q_f32(C + (size_t)(i + 0) * N + j, acc0);
            vst1q_f32(C + (size_t)(i + 1) * N + j, acc1);
            vst1q_f32(C + (size_t)(i + 2) * N + j, acc2);
            vst1q_f32(C + (size_t)(i + 3) * N + j, acc3);
        }

        // Column tail for this 4-row tile.
        for (; j < N; j++) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            const float* a0 = A + (size_t)(i + 0) * K;
            const float* a1 = A + (size_t)(i + 1) * K;
            const float* a2 = A + (size_t)(i + 2) * K;
            const float* a3 = A + (size_t)(i + 3) * K;
            for (k = 0; k < K; k++) {
                float bv = B[(size_t)k * N + j];
                s0 += a0[k] * bv;
                s1 += a1[k] * bv;
                s2 += a2[k] * bv;
                s3 += a3[k] * bv;
            }
            C[(size_t)(i + 0) * N + j] = s0;
            C[(size_t)(i + 1) * N + j] = s1;
            C[(size_t)(i + 2) * N + j] = s2;
            C[(size_t)(i + 3) * N + j] = s3;
        }
    }

    // Row tail.
    for (; i < M; i++) {
        const float* pa = A + (size_t)i * K;
        for (j = 0; j + 4 <= N; j += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            const float* arow = pa;
            const float* pb = B + j;
            for (k = 0; k < K; k++, pb += N) {
                float32x4_t bvec = vld1q_f32(pb);
                acc = vfmaq_n_f32(acc, bvec, *arow++);
            }
            vst1q_f32(C + (size_t)i * N + j, acc);
        }
        for (; j < N; j++) {
            float sum = 0.0f;
            const float* arow = pa;
            const float* pb = B + j;
            for (k = 0; k < K; k++) {
                sum += *arow++ * *pb;
                pb += N;
            }
            C[(size_t)i * N + j] = sum;
        }
    }
}

static void neon_sgemm_rowmajor_packed4(
    const float* __restrict A,
    const float* __restrict packedB,
    const float* __restrict B,
    float* __restrict C,
    int M, int N, int K) {
    int i, j, k;
    int fullN = N & ~3;

    for (i = 0; i + 4 <= M; i += 4) {
        for (j = 0; j < fullN; j += 4) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            const float* pa0 = A + (size_t)(i + 0) * K;
            const float* pa1 = A + (size_t)(i + 1) * K;
            const float* pa2 = A + (size_t)(i + 2) * K;
            const float* pa3 = A + (size_t)(i + 3) * K;
            const float* pb = packedB + (size_t)j * K;

            for (k = 0; k < K; k++, pb += 4) {
                float32x4_t bvec = vld1q_f32(pb);
                acc0 = vfmaq_n_f32(acc0, bvec, *pa0++);
                acc1 = vfmaq_n_f32(acc1, bvec, *pa1++);
                acc2 = vfmaq_n_f32(acc2, bvec, *pa2++);
                acc3 = vfmaq_n_f32(acc3, bvec, *pa3++);
            }

            vst1q_f32(C + (size_t)(i + 0) * N + j, acc0);
            vst1q_f32(C + (size_t)(i + 1) * N + j, acc1);
            vst1q_f32(C + (size_t)(i + 2) * N + j, acc2);
            vst1q_f32(C + (size_t)(i + 3) * N + j, acc3);
        }

        for (; j < N; j++) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            const float* a0 = A + (size_t)(i + 0) * K;
            const float* a1 = A + (size_t)(i + 1) * K;
            const float* a2 = A + (size_t)(i + 2) * K;
            const float* a3 = A + (size_t)(i + 3) * K;
            for (k = 0; k < K; k++) {
                float bv = B[(size_t)k * N + j];
                s0 += a0[k] * bv;
                s1 += a1[k] * bv;
                s2 += a2[k] * bv;
                s3 += a3[k] * bv;
            }
            C[(size_t)(i + 0) * N + j] = s0;
            C[(size_t)(i + 1) * N + j] = s1;
            C[(size_t)(i + 2) * N + j] = s2;
            C[(size_t)(i + 3) * N + j] = s3;
        }
    }

    for (; i < M; i++) {
        const float* pa = A + (size_t)i * K;
        for (j = 0; j < fullN; j += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            const float* arow = pa;
            const float* pb = packedB + (size_t)j * K;
            for (k = 0; k < K; k++, pb += 4) {
                float32x4_t bvec = vld1q_f32(pb);
                acc = vfmaq_n_f32(acc, bvec, *arow++);
            }
            vst1q_f32(C + (size_t)i * N + j, acc);
        }
        for (; j < N; j++) {
            float sum = 0.0f;
            const float* arow = pa;
            const float* pb = B + j;
            for (k = 0; k < K; k++) {
                sum += *arow++ * *pb;
                pb += N;
            }
            C[(size_t)i * N + j] = sum;
        }
    }
}
*/
import "C"

const (
	mulTinyOpsThreshold = 4_096
)

var packedBPool = sync.Pool{
	New: func() any {
		buf := make([]float32, 0)
		return &buf
	},
}

func getPackedB(size int) []float32 {
	if size == 0 {
		return nil
	}
	bufp, _ := packedBPool.Get().(*[]float32)
	if bufp == nil {
		buf := make([]float32, size)
		return buf
	}
	buf := *bufp
	if cap(buf) < size {
		return make([]float32, size)
	}
	return buf[:size]
}

func putPackedB(buf []float32) {
	if buf == nil {
		return
	}
	buf = buf[:0]
	packedBPool.Put(&buf)
}

func mulNeonInto(dst, a, b *Matrix) *Matrix {
	if a.Rows == 0 || b.Cols == 0 || a.Cols == 0 {
		return ensureMatrix(dst, a.Rows, b.Cols)
	}
	if a.Rows*b.Cols*a.Cols <= mulTinyOpsThreshold {
		return mulGenericInto(dst, a, b)
	}
	out := ensureMatrix(dst, a.Rows, b.Cols)
	if len(a.Data) != a.Rows*a.Cols || len(b.Data) != b.Rows*b.Cols || len(out.Data) != out.Rows*out.Cols {
		return mulGenericInto(out, a, b)
	}

	C.neon_sgemm_rowmajor_tiled(
		(*C.float)(unsafe.Pointer(&a.Data[0])),
		(*C.float)(unsafe.Pointer(&b.Data[0])),
		(*C.float)(unsafe.Pointer(&out.Data[0])),
		C.int(a.Rows),
		C.int(b.Cols),
		C.int(a.Cols),
	)
	return out
}

func mulNeonPackedInto(dst, a, b *Matrix) *Matrix {
	out := ensureMatrix(dst, a.Rows, b.Cols)
	if len(a.Data) != a.Rows*a.Cols || len(b.Data) != b.Rows*b.Cols || len(out.Data) != out.Rows*out.Cols {
		return mulGenericInto(out, a, b)
	}

	fullCols := b.Cols &^ 3
	if fullCols == 0 {
		return mulGenericInto(out, a, b)
	}

	packed := getPackedB(a.Cols * fullCols)
	defer putPackedB(packed)
	packB4(packed, b.Data, b.Rows, b.Cols)

	C.neon_sgemm_rowmajor_packed4(
		(*C.float)(unsafe.Pointer(&a.Data[0])),
		(*C.float)(unsafe.Pointer(&packed[0])),
		(*C.float)(unsafe.Pointer(&b.Data[0])),
		(*C.float)(unsafe.Pointer(&out.Data[0])),
		C.int(a.Rows),
		C.int(b.Cols),
		C.int(a.Cols),
	)
	return out
}

func mulNeonPackedIntoMatrix(dst, a *Matrix, b *PackedMatrix) *Matrix {
	if b == nil || b.Original == nil {
		return nil
	}
	out := ensureMatrix(dst, a.Rows, b.Original.Cols)
	if len(a.Data) != a.Rows*a.Cols || len(out.Data) != out.Rows*out.Cols {
		return mulGenericInto(out, a, b.Original)
	}
	if len(b.Data) == 0 || b.FullCols == 0 || len(b.Original.Data) != b.Original.Rows*b.Original.Cols {
		return mulGenericInto(out, a, b.Original)
	}

	C.neon_sgemm_rowmajor_packed4(
		(*C.float)(unsafe.Pointer(&a.Data[0])),
		(*C.float)(unsafe.Pointer(&b.Data[0])),
		(*C.float)(unsafe.Pointer(&b.Original.Data[0])),
		(*C.float)(unsafe.Pointer(&out.Data[0])),
		C.int(a.Rows),
		C.int(b.Original.Cols),
		C.int(a.Cols),
	)
	return out
}
