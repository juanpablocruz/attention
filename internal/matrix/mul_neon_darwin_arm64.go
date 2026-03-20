//go:build darwin && arm64 && cgo

package matrix

/*
#cgo CFLAGS: -O3 -ffast-math
#include <arm_neon.h>

// A: MxK row-major
// B: KxN row-major
// C: MxN row-major
// Tiled 4x4 micro-kernel. We vectorize across 4 output columns at once.
static void neon_sgemm_rowmajor_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
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

            for (k = 0; k < K; k++) {
                float32x4_t bvec = vld1q_f32(B + (size_t)k * N + j);
                acc0 = vfmaq_n_f32(acc0, bvec, a0[k]);
                acc1 = vfmaq_n_f32(acc1, bvec, a1[k]);
                acc2 = vfmaq_n_f32(acc2, bvec, a2[k]);
                acc3 = vfmaq_n_f32(acc3, bvec, a3[k]);
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
        const float* arow = A + (size_t)i * K;
        for (j = 0; j + 4 <= N; j += 4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (k = 0; k < K; k++) {
                float32x4_t bvec = vld1q_f32(B + (size_t)k * N + j);
                acc = vfmaq_n_f32(acc, bvec, arow[k]);
            }
            vst1q_f32(C + (size_t)i * N + j, acc);
        }
        for (; j < N; j++) {
            float sum = 0.0f;
            for (k = 0; k < K; k++) {
                sum += arow[k] * B[(size_t)k * N + j];
            }
            C[(size_t)i * N + j] = sum;
        }
    }
}
*/
import "C"

import "unsafe"

func mulNeon(a, b *Matrix) *Matrix {
	if a.Rows == 0 || b.Cols == 0 || a.Cols == 0 {
		return NewZeroMatrix(a.Rows, b.Cols)
	}
	if len(a.Data) != a.Rows*a.Cols || len(b.Data) != b.Rows*b.Cols {
		return mulGeneric(a, b)
	}
	out := NewZeroMatrix(a.Rows, b.Cols)

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
