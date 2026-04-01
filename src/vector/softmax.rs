// ---------------------------------------------------------------------------
// Compound vector operations -- softmax, normalize
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use super::math;
use super::reduce;

/// In-place softmax: x_i = exp(x_i - max) / sum(exp(x - max))
pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    // 1. find max
    let m = reduce::max(x);

    // 2. subtract max (NEON vectorized)
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let mv = vdupq_n_f32(m);
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                vst1q_f32(x.as_mut_ptr().add(i), vsubq_f32(v, mv));
                i += 4;
            }
        }
    }

    while i < len {
        x[i] -= m;
        i += 1;
    }

    // 3. exp in-place
    math::exp(x);

    // 4. sum
    let s = reduce::sum(x);

    // 5. divide by sum
    if s == 0.0 {
        return;
    }
    let inv_s = 1.0 / s;
    i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let inv_v = vdupq_n_f32(inv_s);
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(v, inv_v));
                i += 4;
            }
        }
    }

    while i < len {
        x[i] *= inv_s;
        i += 1;
    }
}

/// RMS normalization: out_i = (x_i / rms) * weight_i
/// where rms = sqrt(mean(x^2) + eps)
pub fn normalize(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let len = x.len();
    assert_eq!(len, weight.len(), "normalize: x and weight length mismatch");
    assert!(out.len() >= len, "normalize: output buffer too small");

    if len == 0 {
        return;
    }

    // 1. compute mean of squares
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    let ss = {
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        unsafe {
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                acc = vfmaq_f32(acc, v, v);
                i += 4;
            }
        }
        let mut s = unsafe { vaddvq_f32(acc) };
        while i < len {
            s += x[i] * x[i];
            i += 1;
        }
        s
    };

    #[cfg(not(target_arch = "aarch64"))]
    let ss = {
        let mut s = 0.0f32;
        for &v in x {
            s += v * v;
        }
        s
    };

    let rms_inv = 1.0 / (ss / len as f32 + eps).sqrt();

    // 2. normalize and scale by weight
    i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let scale = vdupq_n_f32(rms_inv);
            while i + 4 <= len {
                let vx = vld1q_f32(x.as_ptr().add(i));
                let vw = vld1q_f32(weight.as_ptr().add(i));
                let normed = vmulq_f32(vx, scale);
                let result = vmulq_f32(normed, vw);
                vst1q_f32(out.as_mut_ptr().add(i), result);
                i += 4;
            }
        }
    }

    while i < len {
        out[i] = x[i] * rms_inv * weight[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        softmax(&mut v);
        let s: f32 = v.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
        // should be monotonically increasing
        for i in 1..v.len() {
            assert!(v[i] >= v[i - 1]);
        }
    }

    #[test]
    fn test_softmax_single() {
        let mut v = vec![42.0];
        softmax(&mut v);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_equal() {
        let mut v = vec![1.0; 4];
        softmax(&mut v);
        for &val in &v {
            assert!((val - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_normalize_unit_weight() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let mut out = vec![0.0; 4];
        normalize(&mut out, &x, &w, 1e-5);
        // After normalize with unit weights, the RMS of out should be ~1
        let ss: f32 = out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32;
        assert!((ss - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_normalize_scaling() {
        let x = vec![2.0; 8];
        let w = vec![3.0; 8];
        let mut out = vec![0.0; 8];
        normalize(&mut out, &x, &w, 1e-5);
        // rms = 2.0, so normalized = 1.0, * weight 3.0 = 3.0
        for &v in &out {
            assert!((v - 3.0).abs() < 1e-3);
        }
    }
}
