//! Auto-dispatch GEMM (General Matrix Multiply) routines.
//!
//! All functions compute C += A × B in row-major order:
//!   A is m×k, B is k×n, C is m×n.
//!
//! `sgemm` uses AMX 16×16 microkernel with GEBP cache blocking,
//! parallel across P-cores, falling back to scalar.

mod other;
mod small;

pub use other::{bgemm, hgemm, qgemm};

use crate::matrix;

use std::cell::RefCell;

// ---------------------------------------------------------------------------
// GEBP blocking parameters (tuned for Apple Silicon L1/L2)
// ---------------------------------------------------------------------------

const MR: usize = 16;
const NR: usize = 16;
pub(super) const MC: usize = 64;

// ---------------------------------------------------------------------------
// Thread-local pack buffer cache — keep buffers warm across sgemm calls
// ---------------------------------------------------------------------------

thread_local! {
    static PACK_CACHE: RefCell<PackCache> = const { RefCell::new(PackCache { a: None, b: None }) };
    /// Persistent AMX context — set once per thread, never cleared.
    /// Saves 40ns per sgemm call (AMX_SET + AMX_CLR overhead).
    static AMX_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct PackCache {
    a: Option<AlignedBuf>,
    b: Option<AlignedBuf>,
}

/// Ensure AMX is active on this thread. First call does AMX_SET,
/// subsequent calls are a no-op (Cell::get = single load).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn ensure_amx() {
    AMX_ACTIVE.with(|active| {
        if !active.get() {
            unsafe { matrix::asm::amx_set() };
            active.set(true);
        }
    });
}

/// Get or grow a cached pack buffer. Returns a warm buffer (in L1/L2).
fn cached_buf(slot: &mut Option<AlignedBuf>, needed: usize) -> AlignedBuf {
    if let Some(buf) = slot.take() {
        if buf.len >= needed {
            return buf;
        }
    }
    AlignedBuf::new(needed)
}

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
        let size = n * 4;
        let layout = std::alloc::Layout::from_size_align(size, 64).unwrap();
        // Small buffers: skip zero-fill. Packing writes before reading.
        // Large buffers: zero-fill to pre-fault mmap'd pages.
        let ptr = if size <= 128 * 1024 {
            unsafe { std::alloc::alloc(layout) as *mut f32 }
        } else {
            unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 }
        };
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
        // Tiny matrices (no full tile): NEON, skip AMX entirely.
        if m < MR || n < NR {
            sgemm_neon(a, b, c, m, n, k);
            return;
        }
        // Small matrices: direct AMX, no B packing (stride LDX).
        // B fits in L1 when n*k*4 ≤ 128KB.
        if n * k <= 32768 {
            small::sgemm_amx_direct(a, b, c, m, n, k);
        } else {
            let p_cores = crate::probe::detect().p_cores as usize;
            let max_threads = if m >= MR { m / MR } else { 1 };
            let n_threads = p_cores.max(1).min(max_threads);
            if n_threads > 1 && flops > 1_000_000_000 {
                sgemm_parallel(a, b, c, m, n, k, n_threads);
            } else {
                sgemm_amx_single(a, b, c, m, n, k);
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        sgemm_scalar(a, b, c, m, n, k);
    }
}

/// Parallel sgemm: pre-pack all B, spawn threads once, shared B access.
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
    let kc_max = k.min(if k > 256 { 512 } else { 256 });
    let mc_max = if m > 128 { 256 } else { MC };
    let nc_max = n.min(if n > 256 { 512 } else { 256 });

    let b_block = kc_max * nc_max.div_ceil(NR) * NR;
    let k_blocks = k.div_ceil(kc_max);
    let n_blocks = n.div_ceil(nc_max);

    // Pre-pack ALL B blocks on main thread. Shared across workers.
    let mut b_all = AlignedBuf::new(k_blocks * n_blocks * b_block);
    {
        let mut pc = 0;
        for pi in 0..k_blocks {
            let kc = (k - pc).min(kc_max);
            let mut jc = 0;
            for ji in 0..n_blocks {
                let nc = (n - jc).min(nc_max);
                let off = (pi * n_blocks + ji) * b_block;
                pack_b_nr(b, n, pc, jc, kc, nc, &mut b_all.as_mut_slice()[off..]);
                jc += nc;
            }
            pc += kc;
        }
    }
    let b_packed = b_all.as_slice();

    let base_rows = (m / n_threads / MR) * MR;
    let rows_per_thread = if base_rows == 0 { MR } else { base_rows };

    // Spawn threads ONCE. Each thread handles its M-strip across all (pc, jc).
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
                ensure_amx();

                let mc = m_this.min(mc_max);
                let a_need = mc.div_ceil(MR) * MR * kc_max;
                let mut a_pack = AlignedBuf::new(a_need);

                let mut pc = 0;
                for pi in 0..k_blocks {
                    let kc = (k - pc).min(kc_max);

                    let mut ic = 0;
                    while ic < m_this {
                        let mc_cur = (m_this - ic).min(mc);
                        pack_a_mr(a_slice, k, ic, pc, mc_cur, kc, a_pack.as_mut_slice());

                        let mut jc = 0;
                        for ji in 0..n_blocks {
                            let nc = (n - jc).min(nc_max);
                            let off = (pi * n_blocks + ji) * b_block;
                            gebp_kernel(
                                a_pack.as_slice(),
                                &b_packed[off..],
                                c_chunk,
                                n,
                                ic,
                                jc,
                                mc_cur,
                                nc,
                                kc,
                            );
                            jc += nc;
                        }
                        ic += mc_cur;
                    }
                    pc += kc;
                }
            });

            m_start += m_this;
        }
    });
}

/// Single-threaded AMX sgemm (full GEBP).
#[cfg(target_arch = "aarch64")]
fn sgemm_amx_single(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    ensure_amx();

    // L1 constraint: KC × (MR+NR) × 4 ≤ L1D (64KB).
    // A panel (MC×KC) lives in L2 (4MB). Larger MC = fewer M-panels.
    // Cap to actual dimensions to avoid over-allocation for small matrices.
    let kc_max = k.min(if k > 256 { 512 } else { 256 });
    let mc_max = m.min(if m > 128 { 256 } else { MC }); // MC=256 → A panel in L2
    let nc_max = n.min(if n > 256 { 512 } else { 256 });
    // Packing works in MR/NR-wide strips, so round up allocation.
    let a_need = mc_max.div_ceil(MR) * MR * kc_max;
    let b_need = kc_max * nc_max.div_ceil(NR) * NR;

    // Thread-local cache: reuse warm buffers across calls.
    let (mut a_pack, mut b_pack) = PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let a = cached_buf(&mut cache.a, a_need);
        let b = cached_buf(&mut cache.b, b_need);
        (a, b)
    });

    // Loop order: pc → jc → ic. Pack B once per (pc, jc), reuse across ic.
    // B stays in L2 while A panels are repacked per ic (smaller, NEON-fast).
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

    // Return buffers to thread-local cache for reuse.
    PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        cache.a = Some(a_pack);
        cache.b = Some(b_pack);
    });
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
        // Preload C → AMX compute → store C directly (no AMX→CPU sync).

        // Quad-wide: process 4 B strips at once.
        while jr_strip + 4 <= n_nr && mr_actual == MR && nc - jr_strip * NR >= 4 * NR {
            let jr = jr_strip * NR;
            unsafe {
                let b0 = b_base.add(jr_strip * kc * NR) as *const u8;
                let b1 = b_base.add((jr_strip + 1) * kc * NR) as *const u8;
                let b2 = b_base.add((jr_strip + 2) * kc * NR) as *const u8;
                let b3 = b_base.add((jr_strip + 3) * kc * NR) as *const u8;
                let c_base = c.as_mut_ptr().add((ic + ir) * n + jc + jr);
                for t in 0u8..4 {
                    matrix::tile::preload_c(c_base.add(t as usize * NR), n, t);
                }
                matrix::tile::microkernel_16x64_acc(a_ptr, b0, b1, b2, b3, kc, 64);
                for t in 0u8..4 {
                    matrix::tile::store_c(c_base.add(t as usize * NR), n, t);
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
                let c0 = c.as_mut_ptr().add((ic + ir) * n + jc + jr);
                let c1 = c.as_mut_ptr().add((ic + ir) * n + jc + jr + NR);
                matrix::tile::preload_c(c0, n, 0);
                matrix::tile::preload_c(c1, n, 1);
                matrix::tile::microkernel_16x32_acc(a_ptr, bl, br, kc, 64);
                matrix::tile::store_c(c0, n, 0);
                matrix::tile::store_c(c1, n, 1);
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
                    let cp = c.as_mut_ptr().add((ic + ir) * n + jc + jr);
                    matrix::tile::preload_c(cp, n, 0);
                    matrix::tile::microkernel_16x16_acc(a_ptr, bp, kc, 64);
                    matrix::tile::store_c(cp, n, 0);
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

/// NEON sgemm for tiny matrices (m < MR or n < NR). No AMX.
#[cfg(target_arch = "aarch64")]
fn sgemm_neon(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    use core::arch::aarch64::*;
    unsafe {
        for i in 0..m {
            let mut j = 0;
            while j + 4 <= n {
                let mut acc = vld1q_f32(c.as_ptr().add(i * n + j));
                for p in 0..k {
                    let a_val = vdupq_n_f32(*a.get_unchecked(i * k + p));
                    let b_vec = vld1q_f32(b.as_ptr().add(p * n + j));
                    acc = vfmaq_f32(acc, a_val, b_vec);
                }
                vst1q_f32(c.as_mut_ptr().add(i * n + j), acc);
                j += 4;
            }
            while j < n {
                let mut acc = *c.get_unchecked(i * n + j);
                for p in 0..k {
                    acc += a.get_unchecked(i * k + p) * b.get_unchecked(p * n + j);
                }
                *c.get_unchecked_mut(i * n + j) = acc;
                j += 1;
            }
        }
    }
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
