//go:build darwin && arm64 && cgo

package matrix

/*
#cgo CFLAGS: -O3 -ffast-math
#include <arm_neon.h>
#include <math.h>

static inline float hsumq_f32(float32x4_t v) {
    float32x2_t vlow = vget_low_f32(v);
    float32x2_t vhigh = vget_high_f32(v);
    vlow = vadd_f32(vlow, vhigh);
    float32x2_t sum = vpadd_f32(vlow, vlow);
    return vget_lane_f32(sum, 0);
}

static void neon_add_inplace(float* dst, const float* src, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t dv = vld1q_f32(dst + i);
        float32x4_t sv = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(dv, sv));
    }
    for (; i < n; i++) dst[i] += src[i];
}

static void neon_sub_scaled(float* dst, const float* grad, float scale, int n) {
    float32x4_t sc = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t dv = vld1q_f32(dst + i);
        float32x4_t gv = vld1q_f32(grad + i);
        dv = vsubq_f32(dv, vmulq_f32(sc, gv));
        vst1q_f32(dst + i, dv);
    }
    for (; i < n; i++) dst[i] -= scale * grad[i];
}

static void neon_relu(float* x, int n) {
    float32x4_t z = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vst1q_f32(x + i, vmaxq_f32(v, z));
    }
    for (; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

static float neon_row_max(const float* x, int n) {
    if (n <= 0) return 0.0f;
    int i = 0;
    float32x4_t mv = vdupq_n_f32(x[0]);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        mv = vmaxq_f32(mv, v);
    }
    float lanes[4];
    vst1q_f32(lanes, mv);
    float m = lanes[0];
    if (lanes[1] > m) m = lanes[1];
    if (lanes[2] > m) m = lanes[2];
    if (lanes[3] > m) m = lanes[3];
    for (; i < n; i++) if (x[i] > m) m = x[i];
    return m;
}

static float neon_row_sum_exp_shift(const float* x, int n, float shift) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += expf(x[i] - shift);
    return s;
}

static float neon_row_mean(const float* x, int n) {
    int i = 0;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        acc = vaddq_f32(acc, vld1q_f32(x + i));
    }
    float s = hsumq_f32(acc);
    for (; i < n; i++) s += x[i];
    return s / (float)n;
}

static float neon_row_var(const float* x, int n, float mean) {
    int i = 0;
    float32x4_t acc = vdupq_n_f32(0.0f);
    float32x4_t mv = vdupq_n_f32(mean);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t d = vsubq_f32(v, mv);
        acc = vfmaq_f32(acc, d, d);
    }
    float s = hsumq_f32(acc);
    for (; i < n; i++) {
        float d = x[i] - mean;
        s += d * d;
    }
    return s / (float)n;
}

static void neon_normalize_row(float* dst, const float* src, int n, float mean, float invStd) {
    int i = 0;
    float32x4_t mv = vdupq_n_f32(mean);
    float32x4_t sv = vdupq_n_f32(invStd);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(src + i);
        v = vmulq_f32(vsubq_f32(v, mv), sv);
        vst1q_f32(dst + i, v);
    }
    for (; i < n; i++) dst[i] = (src[i] - mean) * invStd;
}
*/
import "C"

import "unsafe"

func simdAddInPlace(dst, src []float32) {
	if len(dst) == 0 {
		return
	}
	C.neon_add_inplace((*C.float)(unsafe.Pointer(&dst[0])), (*C.float)(unsafe.Pointer(&src[0])), C.int(len(dst)))
}

func simdSubScaled(dst, grad []float32, scale float32) {
	if len(dst) == 0 {
		return
	}
	C.neon_sub_scaled((*C.float)(unsafe.Pointer(&dst[0])), (*C.float)(unsafe.Pointer(&grad[0])), C.float(scale), C.int(len(dst)))
}

func simdReLU(x []float32) {
	if len(x) == 0 {
		return
	}
	C.neon_relu((*C.float)(unsafe.Pointer(&x[0])), C.int(len(x)))
}

func simdRowMax(row []float32) float32 {
	if len(row) == 0 {
		return 0
	}
	return float32(C.neon_row_max((*C.float)(unsafe.Pointer(&row[0])), C.int(len(row))))
}

func simdRowSumExpShift(row []float32, shift float32) float32 {
	if len(row) == 0 {
		return 0
	}
	return float32(C.neon_row_sum_exp_shift((*C.float)(unsafe.Pointer(&row[0])), C.int(len(row)), C.float(shift)))
}

func simdRowMean(row []float32) float32 {
	if len(row) == 0 {
		return 0
	}
	return float32(C.neon_row_mean((*C.float)(unsafe.Pointer(&row[0])), C.int(len(row))))
}

func simdRowVariance(row []float32, mean float32) float32 {
	if len(row) == 0 {
		return 0
	}
	return float32(C.neon_row_var((*C.float)(unsafe.Pointer(&row[0])), C.int(len(row)), C.float(mean)))
}

func simdNormalizeRow(dst, src []float32, mean, invStd float32) {
	if len(src) == 0 {
		return
	}
	C.neon_normalize_row((*C.float)(unsafe.Pointer(&dst[0])), (*C.float)(unsafe.Pointer(&src[0])), C.int(len(src)), C.float(mean), C.float(invStd))
}
