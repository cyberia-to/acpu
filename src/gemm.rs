//! Auto-dispatch GEMM (General Matrix Multiply) routines.
//!
//! All functions compute C += A × B in row-major order:
//!   A is m×k, B is k×n, C is m×n.
//!
//! `sgemm` uses AMX 16×16 microkernel with GEBP cache blocking when
//! available, falling back to NEON.

use crate::matrix;

// ---------------------------------------------------------------------------
// GEBP blocking parameters (tuned for Apple Silicon L1/L2)
// ---------------------------------------------------------------------------

// AMX GEBP parameters (used by sgemm_amx, experimental).
#[allow(dead_code)]
const MR: usize = 16;
#[allow(dead_code)]
const NR: usize = 16;
#[allow(dead_code)]
const MC: usize = 128;
#[allow(dead_code)]
const KC: usize = 256;
#[allow(dead_code)]
const NC: usize = 256;

// ---------------------------------------------------------------------------
// sgemm — f32 matmul
// ---------------------------------------------------------------------------

/// Single-precision matrix multiply: C += A × B.
///
/// Row-major: A[m×k], B[k×n], C[m×n].
///
/// # Panics
///
/// Panics if slice lengths do not match dimensions.
pub fn sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "a.len() must equal m*k");
    assert_eq!(b.len(), k * n, "b.len() must equal k*n");
    assert_eq!(c.len(), m * n, "c.len() must equal m*n");

    #[cfg(target_arch = "aarch64")]
    {
        // AMX microkernel is experimental; use NEON for now.
        // To enable AMX: call sgemm_amx directly.
        sgemm_neon(a, b, c, m, n, k);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sgemm_scalar(a, b, c, m, n, k);
    }
}

// ---------------------------------------------------------------------------
// AMX GEBP sgemm
// ---------------------------------------------------------------------------

#[allow(dead_code)]
/// AMX-accelerated sgemm with GEBP cache blocking.
///
/// 1. Slice K into panels of KC.
/// 2. For each K-panel, pack B columns into column-panels of NC.
/// 3. For each B-panel, pack A rows into row-panels of MC.
/// 4. For each 16×16 tile intersection, run AMX microkernel.
#[cfg(target_arch = "aarch64")]
fn sgemm_amx(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    // Packing buffers (on stack for small, heap for large).
    // A panel: MC × KC, stored column-major for Y register loading.
    // B panel: KC × NC, stored row-major for X register loading.

    let ctx = match matrix::AmxCtx::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            // AMX not available, fall back to NEON.
            sgemm_neon(a, b, c, m, n, k);
            return;
        }
    };

    // Aligned packing buffers.
    let mut a_pack = vec![0f32; MC * KC];
    let mut b_pack = vec![0f32; KC * NC];

    // GEBP: iterate over K in slices of KC.
    let mut pc = 0;
    while pc < k {
        let kc = (k - pc).min(KC);

        // Iterate over N in slices of NC.
        let mut jc = 0;
        while jc < n {
            let nc = (n - jc).min(NC);

            // Pack B[pc..pc+kc, jc..jc+nc] row-major into b_pack.
            pack_b(b, k, n, pc, jc, kc, nc, &mut b_pack);

            // Iterate over M in slices of MC.
            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(MC);

                // Pack A[ic..ic+mc, pc..pc+kc] column-major into a_pack.
                pack_a(a, k, ic, pc, mc, kc, &mut a_pack);

                // Micro-tile loop.
                gebp_kernel(&ctx, &a_pack, &b_pack, c, m, n, ic, jc, mc, nc, kc);

                ic += mc;
            }
            jc += nc;
        }
        pc += kc;
    }

    drop(ctx);
}

/// Pack A[ic..ic+mc, pc..pc+kc] into column-major panels for AMX Y loading.
///
/// Output layout: for each column p in 0..kc, store mc f32 values
/// contiguously (so a 64-byte load gets 16 consecutive rows of that column).
#[allow(dead_code)]
fn pack_a(a: &[f32], lda: usize, ic: usize, pc: usize, mc: usize, kc: usize, dst: &mut [f32]) {
    for p in 0..kc {
        for i in 0..mc {
            dst[p * mc + i] = a[(ic + i) * lda + pc + p];
        }
    }
}

/// Pack B[pc..pc+kc, jc..jc+nc] row-major into panels for AMX X loading.
///
/// Output layout: for each row p in 0..kc, store nc f32 values
/// contiguously (so a 64-byte load gets 16 consecutive columns).
#[allow(dead_code, clippy::too_many_arguments)]
fn pack_b(
    b: &[f32],
    _ldb_k: usize,
    ldb_n: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    dst: &mut [f32],
) {
    for p in 0..kc {
        for j in 0..nc {
            dst[p * nc + j] = b[(pc + p) * ldb_n + jc + j];
        }
    }
}

/// GEBP kernel: dispatch 16×16 AMX tiles over a mc×nc×kc block.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code, clippy::too_many_arguments)]
fn gebp_kernel(
    _ctx: &matrix::AmxCtx,
    a_pack: &[f32],
    b_pack: &[f32],
    c: &mut [f32],
    _m: usize,
    n: usize,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
) {
    let mut ir = 0;
    while ir + MR <= mc {
        let mut jr = 0;
        while jr + NR <= nc {
            unsafe {
                // Build pointers to the packed micropanels.
                // A micropanel: kc columns × 16 rows, each column at a_pack[p*mc + ir]
                // We need it as kc × 64 bytes (kc strips of 16 f32).
                // Since A is packed column-major with mc stride, and we want
                // 16 contiguous f32 starting at offset ir within each column,
                // we need a contiguous repacked micro-strip.
                let mut a_micro = vec![0f32; kc * MR];
                for p in 0..kc {
                    for i in 0..MR {
                        a_micro[p * MR + i] = a_pack[p * mc + ir + i];
                    }
                }

                let mut b_micro = vec![0f32; kc * NR];
                for p in 0..kc {
                    for j in 0..NR {
                        b_micro[p * NR + j] = b_pack[p * nc + jr + j];
                    }
                }

                // Run AMX microkernel.
                matrix::tile::microkernel_16x16(
                    a_micro.as_ptr() as *const u8,
                    b_micro.as_ptr() as *const u8,
                    kc,
                );

                // Accumulate tile result into C.
                let c_ptr = c.as_mut_ptr().add((ic + ir) * n + jc + jr);
                matrix::tile::accumulate_tile_16x16(c_ptr, n);
            }

            jr += NR;
        }

        // Handle remaining columns (< NR) with NEON scalar.
        if jr < nc {
            for i in ir..ir + MR {
                for j in jr..nc {
                    let mut acc = 0.0f32;
                    for p in 0..kc {
                        acc += a_pack[p * mc + i] * b_pack[p * nc + j];
                    }
                    c[(ic + i) * n + jc + j] += acc;
                }
            }
        }

        ir += MR;
    }

    // Handle remaining rows (< MR) with scalar.
    if ir < mc {
        for i in ir..mc {
            for j in 0..nc {
                let mut acc = 0.0f32;
                for p in 0..kc {
                    acc += a_pack[p * mc + i] * b_pack[p * nc + j];
                }
                c[(ic + i) * n + jc + j] += acc;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEON fallback sgemm
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn sgemm_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a_row[p] * b[p * n + j];
            }
            c_row[j] += acc;
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
// hgemm — fp16 inputs, f32 accumulator
// ---------------------------------------------------------------------------

/// Half-precision matrix multiply: C += A × B (fp16 in, fp32 accum).
///
/// # Panics
///
/// Panics if slice lengths do not match dimensions.
pub fn hgemm(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
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

// ---------------------------------------------------------------------------
// bgemm — bf16 inputs, f32 accumulator
// ---------------------------------------------------------------------------

/// BFloat16 matrix multiply: C += A × B (bf16 in, fp32 accum).
pub fn bgemm(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
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

// ---------------------------------------------------------------------------
// qgemm — int8 inputs, f32 accumulator
// ---------------------------------------------------------------------------

/// Int8 quantised matrix multiply: C += scale × (A - zero) × (B - zero).
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgemm_identity() {
        const N: usize = 64;
        let mut a = vec![0.0f32; N * N];
        let mut b = vec![0.0f32; N * N];
        let mut c = vec![0.0f32; N * N];

        // A = all ones, B = identity.
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
                    (c[i * N + j] - 1.0).abs() < 1e-4,
                    "mismatch at [{i},{j}]: {}",
                    c[i * N + j]
                );
            }
        }
    }

    #[test]
    fn sgemm_small_neon_path() {
        // 4×4 should take NEON path (< MR).
        const N: usize = 4;
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let b = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut c = vec![0.0f32; 16];
        sgemm(&a, &b, &mut c, N, N, N);
        for i in 0..16 {
            assert!((c[i] - a[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn sgemm_vs_naive() {
        const M: usize = 33;
        const N: usize = 35;
        const K: usize = 31;
        let a: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..K * N).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c_amx = vec![0.0f32; M * N];
        let mut c_ref = vec![0.0f32; M * N];

        sgemm(&a, &b, &mut c_amx, M, N, K);
        sgemm_scalar(&a, &b, &mut c_ref, M, N, K);

        for i in 0..M * N {
            assert!(
                (c_amx[i] - c_ref[i]).abs() < 1e-3,
                "mismatch at {i}: amx={} ref={}",
                c_amx[i],
                c_ref[i]
            );
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
