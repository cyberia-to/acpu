//! Non-f32 GEMM variants: matmul_f16 (fp16), matmul_bf16 (bf16), matmul_i8 (int8).
//!
//! matmul_f16: AMX FMA16 accelerated (32×32 microkernel, 2048 FLOPs/instruction).
//! matmul_bf16, matmul_i8: scalar reference (AMX MAC16 planned).

use crate::matrix::tile_f16;

const MR_F16: usize = 32;
const NR_F16: usize = 32;

/// Half-precision matrix multiply: C += A × B (fp16 in, fp32 accum, AMX FMA16).
///
/// Uses AMX 32×32 fp16 outer product microkernel. Result accumulated in fp16
/// then converted to f32 on output. 4× theoretical throughput vs sgemm.
///
/// # Panics
/// Panics if slice lengths do not match dimensions.
pub fn matmul_f16(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);

    #[cfg(target_arch = "aarch64")]
    {
        if m >= MR_F16 && n >= NR_F16 {
            matmul_f16_amx(a, b, c, m, n, k);
            return;
        }
    }

    // Scalar fallback for small matrices
    matmul_f16_scalar(a, b, c, m, n, k);
}

fn matmul_f16_scalar(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
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

/// AMX FMA16 hgemm: pack A into 32-wide strips, pack B into 32-wide strips,
/// then GEBP with 32×32 microkernel.
#[cfg(target_arch = "aarch64")]
fn matmul_f16_amx(a: &[u16], b: &[u16], c: &mut [f32], m: usize, n: usize, k: usize) {
    super::ensure_amx();

    // Allocate pack buffers
    let m_strips = m.div_ceil(MR_F16);
    let n_strips = n.div_ceil(NR_F16);
    let a_pack_size = m_strips * MR_F16 * k;
    let b_pack_size = n_strips * NR_F16 * k;
    let mut a_pack = vec![0u16; a_pack_size];
    let mut b_pack = vec![0u16; b_pack_size];

    // Pack A: row-major [m, k] → strips of [MR_F16, k] contiguous
    for strip in 0..m_strips {
        let row_start = strip * MR_F16;
        let rows = MR_F16.min(m - row_start);
        for p in 0..k {
            for r in 0..rows {
                a_pack[strip * MR_F16 * k + p * MR_F16 + r] = a[(row_start + r) * k + p];
            }
            // Zero-pad if rows < MR_F16
            for r in rows..MR_F16 {
                a_pack[strip * MR_F16 * k + p * MR_F16 + r] = 0;
            }
        }
    }

    // Pack B: row-major [k, n] → strips of [NR_F16, k] (transposed: each k-step = 32 contiguous)
    for strip in 0..n_strips {
        let col_start = strip * NR_F16;
        let cols = NR_F16.min(n - col_start);
        for p in 0..k {
            for c_idx in 0..cols {
                b_pack[strip * NR_F16 * k + p * NR_F16 + c_idx] = b[p * n + col_start + c_idx];
            }
            for c_idx in cols..NR_F16 {
                b_pack[strip * NR_F16 * k + p * NR_F16 + c_idx] = 0;
            }
        }
    }

    // GEBP: for each (i_strip, j_strip), run 32×32 microkernel over k
    for i_strip in 0..m_strips {
        let row_start = i_strip * MR_F16;
        let rows = MR_F16.min(m - row_start);
        let a_ptr = a_pack.as_ptr() as *const u8;
        let a_off = i_strip * MR_F16 * k * 2; // bytes

        for j_strip in 0..n_strips {
            let col_start = j_strip * NR_F16;
            let cols = NR_F16.min(n - col_start);
            let b_ptr = b_pack.as_ptr() as *const u8;
            let b_off = j_strip * NR_F16 * k * 2; // bytes

            unsafe {
                // Microkernel: Z[32×32] fp16 = sum_p A[p] ⊗ B[p]
                tile_f16::microkernel_32x32_f16(
                    a_ptr.add(a_off),
                    b_ptr.add(b_off),
                    k,
                );

                // Convert Z fp16 → f32 and accumulate into C
                // Use temp buffer for full 32×32 tile
                let mut tmp_f32 = [0f32; MR_F16 * NR_F16];
                tile_f16::accumulate_tile_f16_to_f32(tmp_f32.as_mut_ptr(), NR_F16);

                // Copy valid portion to C
                for r in 0..rows {
                    for ci in 0..cols {
                        c[(row_start + r) * n + col_start + ci] += tmp_f32[r * NR_F16 + ci];
                    }
                }
            }
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
