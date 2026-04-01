//! Non-f32 GEMM variants: matmul_f16 (fp16), matmul_bf16 (bf16), matmul_i8 (int8).
//!
//! Scalar reference implementations. AMX-accelerated versions planned.

/// Half-precision matrix multiply: C += A × B (fp16 in, fp32 accum).
///
/// # Panics
///
/// Panics if slice lengths do not match dimensions.
pub fn matmul_f16(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let av = crate::numeric::fp16::fp16_to_f32(a[i * k + p]);
                let bv = crate::numeric::fp16::fp16_to_f32(b[p * n + j]);
                acc += av * bv;
            }
            c[i * n + j] += acc;
        }
    }
}

/// BFloat16 matrix multiply: C += A × B (bf16 in, fp32 accum).
pub fn matmul_bf16(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                let av = crate::numeric::bf16::bf16_to_f32(a[i * k + p]);
                let bv = crate::numeric::bf16::bf16_to_f32(b[p * n + j]);
                acc += av * bv;
            }
            c[i * n + j] += acc;
        }
    }
}

/// Int8 quantised matrix multiply: C += scale × (A - zero) × (B - zero).
#[allow(clippy::too_many_arguments)]
pub fn matmul_i8(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    zero: i8,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    let z = zero as i32;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += (a[i * k + p] as i32 - z) * (b[p * n + j] as i32 - z);
            }
            c[i * n + j] += acc as f32 * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_i8_basic() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![1, 0, 0, 1];
        let mut c = vec![0.0f32; 4];
        matmul_i8(&a, &b, &mut c, 2, 2, 2, 1.0, 0);
        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 2.0).abs() < 1e-5);
        assert!((c[2] - 3.0).abs() < 1e-5);
        assert!((c[3] - 4.0).abs() < 1e-5);
    }
}
