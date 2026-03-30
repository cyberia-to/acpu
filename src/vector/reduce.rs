// ---------------------------------------------------------------------------
// Reduction operations -- NEON fast-path + scalar fallback
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Sum of all elements.
pub fn sum(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        unsafe {
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                acc = vaddq_f32(acc, v);
                i += 4;
            }
        }
        let mut s = unsafe { vaddvq_f32(acc) };
        while i < len {
            s += x[i];
            i += 1;
        }
        s
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s = 0.0f32;
        for &v in x {
            s += v;
        }
        s
    }
}

/// Maximum element. Returns -INF for empty slices.
pub fn max(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        let mut acc = unsafe { vdupq_n_f32(f32::NEG_INFINITY) };
        unsafe {
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                acc = vmaxnmq_f32(acc, v);
                i += 4;
            }
        }
        let mut m = unsafe { vmaxnmvq_f32(acc) };
        while i < len {
            m = if x[i] > m { x[i] } else { m };
            i += 1;
        }
        m
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut m = f32::NEG_INFINITY;
        for &v in x {
            if v > m {
                m = v;
            }
        }
        m
    }
}

/// Minimum element. Returns +INF for empty slices.
pub fn min(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return f32::INFINITY;
    }
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        let mut acc = unsafe { vdupq_n_f32(f32::INFINITY) };
        unsafe {
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                acc = vminnmq_f32(acc, v);
                i += 4;
            }
        }
        let mut m = unsafe { vminnmvq_f32(acc) };
        while i < len {
            m = if x[i] < m { x[i] } else { m };
            i += 1;
        }
        m
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut m = f32::INFINITY;
        for &v in x {
            if v < m {
                m = v;
            }
        }
        m
    }
}

/// Dot product of two slices. Panics if lengths differ.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let len = a.len();
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        unsafe {
            while i + 4 <= len {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                acc = vfmaq_f32(acc, va, vb);
                i += 4;
            }
        }
        let mut s = unsafe { vaddvq_f32(acc) };
        while i < len {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s = 0.0f32;
        for j in 0..len {
            s += a[j] * b[j];
        }
        s
    }
}

/// L2 norm: sqrt(sum(x_i^2)).
pub fn norm_l2(x: &[f32]) -> f32 {
    let len = x.len();
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
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
        s.sqrt()
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s = 0.0f32;
        for &v in x {
            s += v * v;
        }
        s.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((sum(&v) - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_max_min() {
        let v = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        assert!((max(&v) - 9.0).abs() < 1e-5);
        assert!((min(&v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l2() {
        let v = vec![3.0, 4.0];
        assert!((norm_l2(&v) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_empty() {
        assert_eq!(sum(&[]), 0.0);
        assert_eq!(max(&[]), f32::NEG_INFINITY);
        assert_eq!(min(&[]), f32::INFINITY);
        assert_eq!(norm_l2(&[]), 0.0);
    }
}
