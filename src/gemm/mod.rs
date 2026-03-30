//! Auto-dispatch GEMM (General Matrix Multiply) routines.
//!
//! All functions compute C += A × B in row-major order:
//!   A is m×k, B is k×n, C is m×n.
//!
//! `sgemm` uses AMX 16×16 microkernel with GEBP cache blocking,
//! parallel across P-cores, falling back to scalar.

mod other;

pub use other::{bgemm, hgemm, qgemm};

use crate::matrix;

// ---------------------------------------------------------------------------
// GEBP blocking parameters (tuned for Apple Silicon L1/L2)
// ---------------------------------------------------------------------------

const MR: usize = 16;
const NR: usize = 16;
pub(super) const MC: usize = 64;

// ---------------------------------------------------------------------------
// Aligned allocation
// ---------------------------------------------------------------------------

pub(super) struct AlignedBuf {
    pub(super) ptr: *mut f32,
    pub(super) len: usize,
}

impl AlignedBuf {
    fn new(n: usize) -> Self {
        if n == 0 {
            return Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            };
        }
        let layout = std::alloc::Layout::from_size_align(n * 4, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 };
        assert!(!ptr.is_null(), "aligned allocation failed");
        Self { ptr, len: n }
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    fn as_slice(&self) -> &[f32] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            let layout = std::alloc::Layout::from_size_align(self.len * 4, 64).unwrap();
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) };
        }
    }
}

// ---------------------------------------------------------------------------
// sgemm — f32 matmul
// ---------------------------------------------------------------------------

/// Single-precision matrix multiply: C += A × B.
///
/// Row-major: A[m×k], B[k×n], C[m×n].
/// Parallelizes across M dimension using P-core threads when beneficial.
pub fn sgemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "a.len() must equal m*k");
    assert_eq!(b.len(), k * n, "b.len() must equal k*n");
    assert_eq!(c.len(), m * n, "c.len() must equal m*n");

    #[cfg(target_arch = "aarch64")]
    {
        let flops = 2 * m * n * k;
        let p_cores = crate::probe::detect().p_cores as usize;
        let max_threads = if m >= MR { m / MR } else { 1 };
        let n_threads = p_cores.max(1).min(max_threads);
        if n_threads > 1 && flops > 16_000_000 {
            sgemm_parallel(a, b, c, m, n, k, n_threads);
        } else {
            sgemm_amx_single(a, b, c, m, n, k);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sgemm_scalar(a, b, c, m, n, k);
    }
}

/// Parallel sgemm: single scope, each thread runs full GEBP on its M-strip.
#[cfg(target_arch = "aarch64")]
fn sgemm_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    n_threads: usize,
) {
    let base = (m / n_threads / MR) * MR;
    let rows_per_thread = if base == 0 { MR } else { base };

    std::thread::scope(|s| {
        let mut c_rest: &mut [f32] = c;
        let mut m_start = 0;

        while m_start < m {
            let m_this = if m - m_start <= rows_per_thread + MR {
                m - m_start
            } else {
                rows_per_thread
            };

            let (c_chunk, rest) = c_rest.split_at_mut(m_this * n);
            c_rest = rest;
            let a_slice = &a[m_start * k..(m_start + m_this) * k];

            s.spawn(move || {
                let _ = crate::sync::affinity::pin_p_core();
                sgemm_amx_single(a_slice, b, c_chunk, m_this, n, k);
            });

            m_start += m_this;
        }
    });
}

/// Single-threaded AMX sgemm (full GEBP).
#[cfg(target_arch = "aarch64")]
fn sgemm_amx_single(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let ctx = match matrix::AmxCtx::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            sgemm_neon(a, b, c, m, n, k);
            return;
        }
    };

    // Adaptive blocking: MC × KC × 4 must fit L1 (128KB).
    // Large KC = fewer K-panels = less packing overhead.
    let (mc_max, kc_max) = if k > 512 && m > 64 {
        (32, 1024) // MC=32 × KC=1024 × 4 = 128KB
    } else {
        (MC, 512) // MC=64 × KC=512 × 4 = 128KB
    };
    let nc_max = if n > 256 { 512 } else { 256 };
    let mut a_pack = AlignedBuf::new(mc_max * kc_max);
    let mut b_pack = AlignedBuf::new(kc_max * nc_max);

    let mut pc = 0;
    while pc < k {
        let kc = (k - pc).min(kc_max);

        let mut jc = 0;
        while jc < n {
            let nc = (n - jc).min(nc_max);
            pack_b_nr(b, n, pc, jc, kc, nc, b_pack.as_mut_slice());

            let mut ic = 0;
            while ic < m {
                let mc = (m - ic).min(mc_max);
                pack_a_mr(a, k, ic, pc, mc, kc, a_pack.as_mut_slice());
                gebp_kernel(
                    &ctx,
                    a_pack.as_slice(),
                    b_pack.as_slice(),
                    c,
                    n,
                    ic,
                    jc,
                    mc,
                    nc,
                    kc,
                );
                ic += mc;
            }
            jc += nc;
        }
        pc += kc;
    }

    drop(ctx);
}

// ---------------------------------------------------------------------------
// Packing: direct MR/NR-width strips (no repack needed)
// ---------------------------------------------------------------------------

/// Pack A[ic..ic+mc, pc..pc+kc] into MR-wide contiguous strips.
///
/// Layout: n_strips × kc × MR, where each strip is MR contiguous f32
/// per k step. Microkernel reads directly without repacking.
pub(super) fn pack_a_mr(
    a: &[f32],
    lda: usize,
    ic: usize,
    pc: usize,
    mc: usize,
    kc: usize,
    dst: &mut [f32],
) {
    let n_full = mc / MR;
    let rem = mc % MR;

    for s in 0..n_full {
        let base = s * kc * MR;
        let row_start = ic + s * MR;

        #[cfg(target_arch = "aarch64")]
        {
            pack_a_strip_neon(a, lda, row_start, pc, kc, &mut dst[base..]);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..MR {
                let a_row = (row_start + i) * lda + pc;
                for p in 0..kc {
                    dst[base + p * MR + i] = a[a_row + p];
                }
            }
        }
    }

    if rem > 0 {
        let base = n_full * kc * MR;
        let row_start = ic + n_full * MR;
        // Zero the full strip first, then overwrite valid rows.
        for p in 0..kc {
            for i in 0..MR {
                dst[base + p * MR + i] = 0.0;
            }
        }
        for i in 0..rem {
            let a_row = (row_start + i) * lda + pc;
            for p in 0..kc {
                dst[base + p * MR + i] = a[a_row + p];
            }
        }
    }
}

/// NEON-accelerated pack: 4×4 transpose blocks for one MR-wide strip.
#[cfg(target_arch = "aarch64")]
fn pack_a_strip_neon(
    a: &[f32],
    lda: usize,
    row_start: usize,
    pc: usize,
    kc: usize,
    dst: &mut [f32],
) {
    use core::arch::aarch64::*;

    // Process 4 rows × 4 columns at a time using NEON transpose.
    // MR=16 → 4 groups of 4 rows. kc → kc/4 groups of 4 columns.
    for ig in 0..(MR / 4) {
        let i = ig * 4;
        let a0 = (row_start + i) * lda + pc;
        let a1 = (row_start + i + 1) * lda + pc;
        let a2 = (row_start + i + 2) * lda + pc;
        let a3 = (row_start + i + 3) * lda + pc;

        let mut p = 0;
        while p + 4 <= kc {
            unsafe {
                let r0 = vld1q_f32(a.as_ptr().add(a0 + p));
                let r1 = vld1q_f32(a.as_ptr().add(a1 + p));
                let r2 = vld1q_f32(a.as_ptr().add(a2 + p));
                let r3 = vld1q_f32(a.as_ptr().add(a3 + p));

                // 4×4 transpose via zip1/zip2 at f32 and f64 granularity.
                let lo01 = vzip1q_f32(r0, r1);
                let hi01 = vzip2q_f32(r0, r1);
                let lo23 = vzip1q_f32(r2, r3);
                let hi23 = vzip2q_f32(r2, r3);

                let c0 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(lo01),
                    vreinterpretq_f64_f32(lo23),
                ));
                let c1 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(lo01),
                    vreinterpretq_f64_f32(lo23),
                ));
                let c2 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(hi01),
                    vreinterpretq_f64_f32(hi23),
                ));
                let c3 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(hi01),
                    vreinterpretq_f64_f32(hi23),
                ));

                let d = dst.as_mut_ptr();
                vst1q_f32(d.add(p * MR + i), c0);
                vst1q_f32(d.add((p + 1) * MR + i), c1);
                vst1q_f32(d.add((p + 2) * MR + i), c2);
                vst1q_f32(d.add((p + 3) * MR + i), c3);
            }
            p += 4;
        }

        // Remainder columns: scalar.
        while p < kc {
            dst[p * MR + i] = a[a0 + p];
            dst[p * MR + i + 1] = a[a1 + p];
            dst[p * MR + i + 2] = a[a2 + p];
            dst[p * MR + i + 3] = a[a3 + p];
            p += 1;
        }
    }
}

/// Pack B[pc..pc+kc, jc..jc+nc] into NR-wide contiguous strips.
pub(super) fn pack_b_nr(
    b: &[f32],
    ldb: usize,
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    dst: &mut [f32],
) {
    let n_full = nc / NR;
    let rem = nc % NR;

    for s in 0..n_full {
        let base = s * kc * NR;
        let col_start = jc + s * NR;
        for p in 0..kc {
            let src_row = (pc + p) * ldb + col_start;
            let dst_off = base + p * NR;
            dst[dst_off..dst_off + NR].copy_from_slice(&b[src_row..src_row + NR]);
        }
    }

    if rem > 0 {
        let base = n_full * kc * NR;
        let col_start = jc + n_full * NR;
        for p in 0..kc {
            let src_row = (pc + p) * ldb + col_start;
            let dst_off = base + p * NR;
            dst[dst_off..dst_off + rem].copy_from_slice(&b[src_row..src_row + rem]);
            for j in rem..NR {
                dst[dst_off + j] = 0.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GEBP kernel: direct micropanel pointers (no repack)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(super) fn gebp_kernel(
    _ctx: &matrix::AmxCtx,
    a_pack: &[f32],
    b_pack: &[f32],
    c: &mut [f32],
    n: usize,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
) {
    let n_mr = mc.div_ceil(MR);
    let n_nr = nc.div_ceil(NR);

    for ir_strip in 0..n_mr {
        let ir = ir_strip * MR;
        let mr_actual = MR.min(mc - ir);
        let a_ptr = unsafe { a_pack.as_ptr().add(ir_strip * kc * MR) as *const u8 };

        let mut jr_strip = 0;
        let b_base = b_pack.as_ptr();

        // Cascade: 16×64 (4 tiles) → 16×32 (2 tiles) → 16×16 (1 tile).
        // Quad-wide: process 4 B strips at once.
        while jr_strip + 4 <= n_nr && mr_actual == MR && nc - jr_strip * NR >= 4 * NR {
            let jr = jr_strip * NR;
            unsafe {
                let b0 = b_base.add(jr_strip * kc * NR) as *const u8;
                let b1 = b_base.add((jr_strip + 1) * kc * NR) as *const u8;
                let b2 = b_base.add((jr_strip + 2) * kc * NR) as *const u8;
                let b3 = b_base.add((jr_strip + 3) * kc * NR) as *const u8;
                matrix::tile::microkernel_16x64(a_ptr, b0, b1, b2, b3, kc);
                for t in 0u8..4 {
                    let cp = c
                        .as_mut_ptr()
                        .add((ic + ir) * n + jc + jr + t as usize * NR);
                    matrix::tile::accumulate_tile(cp, n, t);
                }
            }
            jr_strip += 4;
        }

        // Double-wide: process 2 B strips.
        while jr_strip + 2 <= n_nr && mr_actual == MR && nc - jr_strip * NR >= 2 * NR {
            let jr = jr_strip * NR;
            unsafe {
                let bl = b_base.add(jr_strip * kc * NR) as *const u8;
                let br = b_base.add((jr_strip + 1) * kc * NR) as *const u8;
                matrix::tile::microkernel_16x32(a_ptr, bl, br, kc);
                let c0 = c.as_mut_ptr().add((ic + ir) * n + jc + jr);
                matrix::tile::accumulate_tile(c0, n, 0);
                let c1 = c.as_mut_ptr().add((ic + ir) * n + jc + jr + NR);
                matrix::tile::accumulate_tile(c1, n, 1);
            }
            jr_strip += 2;
        }

        // Single tiles (including edge cases).
        while jr_strip < n_nr {
            let jr = jr_strip * NR;
            let nr_actual = NR.min(nc - jr);
            if mr_actual == MR && nr_actual == NR {
                unsafe {
                    let bp = b_base.add(jr_strip * kc * NR) as *const u8;
                    matrix::tile::microkernel_16x16(a_ptr, bp, kc);
                    let cp = c.as_mut_ptr().add((ic + ir) * n + jc + jr);
                    matrix::tile::accumulate_tile_16x16(cp, n);
                }
            } else {
                edge_kernel(
                    a_pack, b_pack, c, n, ic, jc, ir_strip, jr_strip, mr_actual, nr_actual, kc,
                );
            }
            jr_strip += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn edge_kernel(
    a_pack: &[f32],
    b_pack: &[f32],
    c: &mut [f32],
    n: usize,
    ic: usize,
    jc: usize,
    ir_strip: usize,
    jr_strip: usize,
    mr: usize,
    nr: usize,
    kc: usize,
) {
    let ir = ir_strip * MR;
    let jr = jr_strip * NR;
    for i in 0..mr {
        for j in 0..nr {
            let mut acc = 0.0f32;
            for p in 0..kc {
                let av = a_pack[ir_strip * kc * MR + p * MR + i];
                let bv = b_pack[jr_strip * kc * NR + p * NR + j];
                acc += av * bv;
            }
            c[(ic + ir + i) * n + jc + jr + j] += acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Fallbacks
// ---------------------------------------------------------------------------

/// NEON-tiled sgemm for small matrices. No packing, no AMX, zero overhead.
/// Uses 4-wide NEON fmla directly on input data.
#[cfg(target_arch = "aarch64")]
fn sgemm_neon_tiled(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use core::arch::aarch64::*;
    unsafe {
        let mut i = 0;
        while i < m {
            let mut j = 0;
            while j + 4 <= n {
                // Accumulate 1×4 block of C.
                let mut acc = vld1q_f32(c.as_ptr().add(i * n + j));
                for p in 0..k {
                    let a_val = vdupq_n_f32(*a.get_unchecked(i * k + p));
                    let b_vec = vld1q_f32(b.as_ptr().add(p * n + j));
                    acc = vfmaq_f32(acc, a_val, b_vec);
                }
                vst1q_f32(c.as_mut_ptr().add(i * n + j), acc);
                j += 4;
            }
            // Remainder columns.
            while j < n {
                let mut acc = *c.get_unchecked(i * n + j);
                for p in 0..k {
                    acc += a.get_unchecked(i * k + p) * b.get_unchecked(p * n + j);
                }
                *c.get_unchecked_mut(i * n + j) = acc;
                j += 1;
            }
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn sgemm_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    sgemm_neon_tiled(a, b, c, m, n, k);
}

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
    fn sgemm_small() {
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
    fn sgemm_large_parallel() {
        const M: usize = 256;
        const N: usize = 256;
        const K: usize = 256;
        let a: Vec<f32> = (0..M * K).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let b: Vec<f32> = (0..K * N).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let mut c_par = vec![0.0f32; M * N];
        let mut c_ref = vec![0.0f32; M * N];

        sgemm(&a, &b, &mut c_par, M, N, K);
        sgemm_scalar(&a, &b, &mut c_ref, M, N, K);

        let mut max_err = 0.0f32;
        for i in 0..M * N {
            let err = (c_par[i] - c_ref[i]).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(max_err < 0.1, "parallel sgemm 256x256: max_err={max_err}");
    }
}
