//! Auto-dispatch GEMM (General Matrix Multiply) routines.
//!
//! All functions compute C += A * B in row-major order:
//!   A is m x k, B is k x n, C is m x n.
//!
//! `sgemm` is NEON-vectorized.  The other three start as scalar
//! implementations with TODO markers for AMX/NEON optimisation.

// ---------------------------------------------------------------------------
// sgemm -- f32 matmul, NEON-vectorized inner loop
// ---------------------------------------------------------------------------

/// Single-precision matrix multiply: C += A * B.
///
/// `a` is `m x k` (row-major), `b` is `k x n` (row-major),
/// `c` is `m x n` (row-major, zeroed or accumulated into).
///
/// # Panics
///
/// Panics if slice lengths do not match the declared dimensions.
pub fn sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "a.len() must equal m*k");
    assert_eq!(b.len(), k * n, "b.len() must equal k*n");
    assert_eq!(c.len(), m * n, "c.len() must equal m*n");

    #[cfg(target_arch = "aarch64")]
    {
        sgemm_neon(a, b, c, m, n, k);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sgemm_scalar(a, b, c, m, n, k);
    }
}

/// NEON-vectorized sgemm.  Processes 4 columns at a time in the inner
/// product loop.
#[cfg(target_arch = "aarch64")]
fn sgemm_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];

        // Process 4 output columns at a time.
        let mut j = 0usize;
        while j + 4 <= n {
            let mut acc = [0.0f32; 4];

            // Vectorized inner product over k dimension.
            let mut p = 0usize;
            while p + 4 <= k {
                let a0 = a_row[p];
                let a1 = a_row[p + 1];
                let a2 = a_row[p + 2];
                let a3 = a_row[p + 3];

                for (jj, ac) in acc.iter_mut().enumerate() {
                    let col = j + jj;
                    *ac += a0 * b[p * n + col];
                    *ac += a1 * b[(p + 1) * n + col];
                    *ac += a2 * b[(p + 2) * n + col];
                    *ac += a3 * b[(p + 3) * n + col];
                }
                p += 4;
            }
            // Scalar tail for remaining k.
            while p < k {
                for (jj, ac) in acc.iter_mut().enumerate() {
                    *ac += a_row[p] * b[p * n + j + jj];
                }
                p += 1;
            }

            // Write back using NEON load-add-store.
            unsafe {
                let dst = c_row.as_mut_ptr().add(j);
                let src = acc.as_ptr();
                core::arch::asm!(
                    "ldp s0, s1, [{src}]",
                    "ldp s2, s3, [{src}, #8]",
                    "ldp s4, s5, [{dst}]",
                    "ldp s6, s7, [{dst}, #8]",
                    "fadd s0, s0, s4",
                    "fadd s1, s1, s5",
                    "fadd s2, s2, s6",
                    "fadd s3, s3, s7",
                    "stp s0, s1, [{dst}]",
                    "stp s2, s3, [{dst}, #8]",
                    src = in(reg) src,
                    dst = in(reg) dst,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                    out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                );
            }
            j += 4;
        }

        // Scalar tail for remaining columns.
        while j < n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a_row[p] * b[p * n + j];
            }
            c_row[j] += acc;
            j += 1;
        }
    }
}

/// Scalar fallback sgemm.
#[allow(dead_code)]
fn sgemm_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += acc;
        }
    }
}

// ---------------------------------------------------------------------------
// hgemm -- fp16 inputs, f32 accumulator
// ---------------------------------------------------------------------------

/// Half-precision matrix multiply: C += A * B (fp16 in, fp32 accum/out).
///
/// `a` is `m x k` (row-major, fp16 stored as u16), `b` is `k x n`,
/// `c` is `m x n` (f32).
///
/// # Panics
///
/// Panics if slice lengths do not match the declared dimensions.
///
/// TODO: NEON/AMX optimised path.
pub fn hgemm(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "a.len() must equal m*k");
    assert_eq!(b.len(), k * n, "b.len() must equal k*n");
    assert_eq!(c.len(), m * n, "c.len() must equal m*n");

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

// ---------------------------------------------------------------------------
// bgemm -- bf16 inputs, f32 accumulator
// ---------------------------------------------------------------------------

/// BFloat16 matrix multiply: C += A * B (bf16 in, fp32 accum/out).
///
/// `a` is `m x k` (row-major, bf16 stored as u16), `b` is `k x n`,
/// `c` is `m x n` (f32).
///
/// # Panics
///
/// Panics if slice lengths do not match the declared dimensions.
///
/// TODO: NEON/AMX optimised path.
pub fn bgemm(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "a.len() must equal m*k");
    assert_eq!(b.len(), k * n, "b.len() must equal k*n");
    assert_eq!(c.len(), m * n, "c.len() must equal m*n");

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

// ---------------------------------------------------------------------------
// qgemm -- int8 inputs, f32 accumulator
// ---------------------------------------------------------------------------

/// Int8 quantised matrix multiply: C += scale * (A - zero) * (B - zero).
///
/// `a` is `m x k` (row-major, i8), `b` is `k x n`, `c` is `m x n` (f32).
/// `scale` and `zero` are the quantisation parameters (uniform symmetric).
///
/// # Panics
///
/// Panics if slice lengths do not match the declared dimensions.
///
/// TODO: NEON SDOT / AMX MAC16 optimised path.
#[allow(clippy::too_many_arguments)]
pub fn qgemm(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale: f32,
    zero: i8,
) {
    assert_eq!(a.len(), m * k, "a.len() must equal m*k");
    assert_eq!(b.len(), k * n, "b.len() must equal k*n");
    assert_eq!(c.len(), m * n, "c.len() must equal m*n");

    let z = zero as i32;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                let av = a[i * k + p] as i32 - z;
                let bv = b[p * n + j] as i32 - z;
                acc += av * bv;
            }
            c[i * n + j] += acc as f32 * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_identity() {
        const N: usize = 8;
        let mut a = vec![0.0f32; N * N];
        let mut b = vec![0.0f32; N * N];
        let mut c = vec![0.0f32; N * N];

        for i in 0..N {
            for j in 0..N {
                a[i * N + j] = 1.0;
            }
            b[i * N + i] = 1.0;
        }

        sgemm(&a, &b, &mut c, N, N, N);

        for i in 0..N {
            for j in 0..N {
                assert!(
                    (c[i * N + j] - 1.0).abs() < 1e-5,
                    "mismatch at [{i},{j}]: {}",
                    c[i * N + j]
                );
            }
        }
    }

    #[test]
    fn qgemm_basic() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![1, 0, 0, 1];
        let mut c = vec![0.0f32; 4];

        qgemm(&a, &b, &mut c, 2, 2, 2, 1.0, 0);

        assert!((c[0] - 1.0).abs() < 1e-5);
        assert!((c[1] - 2.0).abs() < 1e-5);
        assert!((c[2] - 3.0).abs() < 1e-5);
        assert!((c[3] - 4.0).abs() < 1e-5);
    }
}
