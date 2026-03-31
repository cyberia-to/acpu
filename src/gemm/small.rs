//! Direct AMX path for small matrices.
//! 32×32 pair-load kernel when m,n both ≥32 and divisible by 32.
//! 16×64 fallback for all other cases.

use super::{MR, NR};
use crate::matrix::tile;

/// Direct AMX sgemm for small matrices where n*k ≤ 32K.
#[cfg(target_arch = "aarch64")]
pub(super) fn sgemm_amx_direct(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    super::ensure_amx();

    let n_mr = m.div_ceil(MR);
    let bs = n * 4;
    let n_nr = n.div_ceil(NR);

    // Use 32×32 pair-load path when both dimensions are multiples of 32.
    let use_pair = m % 32 == 0 && n % 32 == 0 && m >= 32 && n >= 32;

    if use_pair {
        sgemm_pair32(a, b, c, m, n, k, n_mr, n_nr, bs);
    } else {
        sgemm_16x64(a, b, c, m, n, k, n_mr, n_nr, bs);
    }
}

/// 32×32 pair-load path: interleaved A pack + pair LDY.
/// 7 ops/k-step (1 LDY pair + 2 LDX + 4 FMA) vs 9 in 16×64.
#[cfg(target_arch = "aarch64")]
fn sgemm_pair32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    n_mr: usize,
    n_nr: usize,
    bs: usize,
) {
    let n_pairs = n_mr / 2;
    let a_need = n_pairs * k * MR * 2;
    let mut a_pack = super::PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        super::cached_buf(&mut cache.a, a_need)
    });

    // Pack all pairs upfront, then compute.
    let dst = a_pack.as_mut_slice();
    for pair in 0..n_pairs {
        let off = pair * k * MR * 2;
        pack_a_interleaved_neon(a, k, pair * 2 * MR, k, &mut dst[off..]);
    }

    unsafe {
        let a_base = a_pack.as_slice().as_ptr();
        let b_base = b.as_ptr();

        for pair in 0..n_pairs {
            let ap = a_base.add(pair * k * MR * 2) as *const u8;
            let ir = pair * 2;
            let mut jr = 0usize;

            while jr + 2 <= n_nr {
                let c0 = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                tile::preload_c(c0, n, 0);
                tile::preload_c(c0.add(NR), n, 1);
                tile::preload_c(c0.add(MR * n), n, 2);
                tile::preload_c(c0.add(MR * n + NR), n, 3);
                tile::microkernel_32x32_acc(
                    ap,
                    ap,
                    b_base.add(jr * NR) as *const u8,
                    std::ptr::null(),
                    k,
                    bs,
                );
                tile::store_c(c0, n, 0);
                tile::store_c(c0.add(NR), n, 1);
                tile::store_c(c0.add(MR * n), n, 2);
                tile::store_c(c0.add(MR * n + NR), n, 3);
                jr += 2;
            }
        }
    }

    super::PACK_CACHE.with(|c| {
        c.borrow_mut().a = Some(a_pack);
    });
}

/// NEON-accelerated interleaved A pack for 2 strips (32 rows).
/// Output stride = 32 floats (128 bytes) per k column for pair LDY.
#[cfg(target_arch = "aarch64")]
pub(super) fn pack_a_interleaved_neon(
    a: &[f32],
    lda: usize,
    row_start: usize,
    kc: usize,
    dst: &mut [f32],
) {
    use core::arch::aarch64::*;

    let mut rows = [0usize; 32];
    for (i, row) in rows.iter_mut().enumerate() {
        *row = (row_start + i) * lda;
    }

    let mut p = 0;
    while p + 4 <= kc {
        unsafe {
            let d = dst.as_mut_ptr();
            let ap = a.as_ptr();
            for ig in 0..8u32 {
                let i = (ig * 4) as usize;
                let r0 = vld1q_f32(ap.add(rows[i] + p));
                let r1 = vld1q_f32(ap.add(rows[i + 1] + p));
                let r2 = vld1q_f32(ap.add(rows[i + 2] + p));
                let r3 = vld1q_f32(ap.add(rows[i + 3] + p));
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
                // Interleaved stride: 32 floats per k-step.
                vst1q_f32(d.add(p * 32 + i), c0);
                vst1q_f32(d.add((p + 1) * 32 + i), c1);
                vst1q_f32(d.add((p + 2) * 32 + i), c2);
                vst1q_f32(d.add((p + 3) * 32 + i), c3);
            }
        }
        p += 4;
    }
    while p < kc {
        for i in 0..32 {
            dst[p * 32 + i] = a[rows[i] + p];
        }
        p += 1;
    }
}

/// Original 16×64 path with interleaved pack/compute.
#[cfg(target_arch = "aarch64")]
fn sgemm_16x64(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    n_mr: usize,
    n_nr: usize,
    bs: usize,
) {
    let a_need = n_mr * MR * k;
    let mut a_pack = super::PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        super::cached_buf(&mut cache.a, a_need)
    });

    let first_mc = MR.min(m);
    super::pack_a_mr(a, k, 0, 0, first_mc, k, a_pack.as_mut_slice());

    unsafe {
        for ir in 0..n_mr {
            let mr = MR.min(m - ir * MR);
            let ap = a_pack.as_slice().as_ptr().add(ir * k * MR) as *const u8;
            let mut jr = 0usize;

            while jr + 4 <= n_nr && mr == MR && (n - jr * NR) >= 4 * NR {
                let cp = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                for t in 0u8..4 {
                    tile::preload_c(cp.add(t as usize * NR), n, t);
                }
                tile::microkernel_16x64_acc(
                    ap,
                    b.as_ptr().add(jr * NR) as *const u8,
                    b.as_ptr().add((jr + 1) * NR) as *const u8,
                    b.as_ptr().add((jr + 2) * NR) as *const u8,
                    b.as_ptr().add((jr + 3) * NR) as *const u8,
                    k,
                    bs,
                );
                for t in 0u8..4 {
                    tile::store_c(cp.add(t as usize * NR), n, t);
                }
                jr += 4;
            }
            while jr + 2 <= n_nr && mr == MR && (n - jr * NR) >= 2 * NR {
                let cp = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                tile::preload_c(cp, n, 0);
                tile::preload_c(cp.add(NR), n, 1);
                tile::microkernel_16x32_acc(
                    ap,
                    b.as_ptr().add(jr * NR) as *const u8,
                    b.as_ptr().add((jr + 1) * NR) as *const u8,
                    k,
                    bs,
                );
                tile::store_c(cp, n, 0);
                tile::store_c(cp.add(NR), n, 1);
                jr += 2;
            }
            while jr < n_nr {
                let nr = NR.min(n - jr * NR);
                if mr == MR && nr == NR {
                    let cp = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                    tile::preload_c(cp, n, 0);
                    tile::microkernel_16x16_acc(ap, b.as_ptr().add(jr * NR) as *const u8, k, bs);
                    tile::store_c(cp, n, 0);
                } else {
                    for i in 0..mr {
                        for j in 0..nr {
                            let mut acc = 0.0f32;
                            for p in 0..k {
                                acc += a[(ir * MR + i) * k + p] * b[p * n + jr * NR + j];
                            }
                            c[(ir * MR + i) * n + jr * NR + j] += acc;
                        }
                    }
                }
                jr += 1;
            }

            if ir + 1 < n_mr {
                let next_mc = MR.min(m - (ir + 1) * MR);
                let off = (ir + 1) * k * MR;
                super::pack_a_mr(
                    a,
                    k,
                    (ir + 1) * MR,
                    0,
                    next_mc,
                    k,
                    &mut a_pack.as_mut_slice()[off..],
                );
            }
        }
    }

    super::PACK_CACHE.with(|c| {
        c.borrow_mut().a = Some(a_pack);
    });
}
