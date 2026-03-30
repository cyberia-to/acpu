//! Complex multiply-accumulate using FCMLA (ARMv8.3-A).
//!
//! TODO: Implement NEON FCMLA vectorized path.

/// Complex multiply-accumulate: `acc += a * b` (element-wise complex).
///
/// Inputs are interleaved complex: `[re0, im0, re1, im1, ...]`.
/// Each slice length must be even (pairs of `(re, im)`).
///
/// Uses FCMLA on aarch64 when available (ARMv8.3-A FEAT_FCMA).
///
/// # Panics
///
/// Panics if any slice has odd length or if `a` and `b` differ in length.
pub fn complex_mul_acc(acc: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "a and b must have the same length");
    assert_eq!(a.len() % 2, 0, "complex slices must have even length");
    let n = acc.len().min(a.len());
    assert_eq!(n % 2, 0, "acc length must be even");

    // TODO: aarch64 FCMLA path
    //
    // The FCMLA instruction performs fused complex multiply-accumulate:
    //   fcmla v_acc.4s, v_a.4s, v_b.4s, #0   -- rotate by 0
    //   fcmla v_acc.4s, v_a.4s, v_b.4s, #90   -- rotate by 90
    //
    // Together these compute:
    //   acc.re += a.re * b.re - a.im * b.im
    //   acc.im += a.re * b.im + a.im * b.re
    //
    // Requires FEAT_FCMA (Apple M1+, Cortex-A75+).

    // Scalar fallback
    let mut i = 0;
    while i + 1 < n {
        let ar = a[i];
        let ai = a[i + 1];
        let br = b[i];
        let bi = b[i + 1];
        acc[i] += ar * br - ai * bi;
        acc[i + 1] += ar * bi + ai * br;
        i += 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_mul_acc_basic() {
        // (1+2i) * (3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        let mut acc = [0.0f32, 0.0];
        complex_mul_acc(&mut acc, &a, &b);
        assert!((acc[0] - (-5.0)).abs() < 1e-6);
        assert!((acc[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn complex_mul_acc_accumulates() {
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut acc = [10.0f32, 20.0, 30.0, 40.0];
        complex_mul_acc(&mut acc, &a, &b);
        // (1+0i)*(1+0i) = 1+0i => acc[0..2] = [11, 20]
        // (0+1i)*(0+1i) = -1+0i => acc[2..4] = [29, 40]
        assert!((acc[0] - 11.0).abs() < 1e-6);
        assert!((acc[1] - 20.0).abs() < 1e-6);
        assert!((acc[2] - 29.0).abs() < 1e-6);
        assert!((acc[3] - 40.0).abs() < 1e-6);
    }
}
