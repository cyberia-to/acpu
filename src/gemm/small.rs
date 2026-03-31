//! Direct AMX path for small matrices. Zero heap allocation.
//! Preload C → AMX compute → store C directly (no AMX→CPU sync).

use super::{MR, NR};
use crate::matrix;
use crate::matrix::tile;
use core::mem::MaybeUninit;

/// Direct AMX sgemm for small matrices (m≤128, n≤128, k≤512).
/// Zero heap alloc. Preload C into Z, accumulate, store directly.
#[cfg(target_arch = "aarch64")]
pub(super) fn sgemm_amx_direct(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let ctx = match matrix::AmxCtx::new() {
        Ok(c) => c,
        Err(_) => return,
    };

    // Pack A on stack using MaybeUninit — no zero-fill overhead.
    #[repr(align(64))]
    struct APack([MaybeUninit<f32>; 128 * 512]);

    let mut a_buf: APack = unsafe { MaybeUninit::uninit().assume_init() };
    let a_pack = &mut a_buf.0[..];

    let n_mr = m.div_ceil(MR);
    for s in 0..n_mr {
        let rs = s * MR;
        let rows = MR.min(m - rs);
        let base = s * k * MR;
        for i in 0..rows {
            let a_row = (rs + i) * k;
            for p in 0..k {
                a_pack[base + p * MR + i] = MaybeUninit::new(a[a_row + p]);
            }
        }
        for i in rows..MR {
            for p in 0..k {
                a_pack[base + p * MR + i] = MaybeUninit::new(0.0);
            }
        }
    }

    // Pack B into NR-wide strips on stack (needed for microkernel alignment).
    #[repr(align(64))]
    struct BPack([MaybeUninit<f32>; 128 * 512]);

    let mut b_buf: BPack = unsafe { MaybeUninit::uninit().assume_init() };
    let b_pack = &mut b_buf.0[..];

    let n_nr = n.div_ceil(NR);
    for s in 0..n_nr {
        let cs = s * NR;
        let cols = NR.min(n - cs);
        let base = s * k * NR;
        for p in 0..k {
            let src = p * n + cs;
            let dst = base + p * NR;
            for j in 0..cols {
                b_pack[dst + j] = MaybeUninit::new(b[src + j]);
            }
            for j in cols..NR {
                b_pack[dst + j] = MaybeUninit::new(0.0);
            }
        }
    }

    unsafe {
        for ir in 0..n_mr {
            let mr = MR.min(m - ir * MR);
            let ap = a_pack.as_ptr().add(ir * k * MR) as *const u8;

            // Quad-wide: 4 B strips at once.
            let mut jr = 0usize;
            while jr + 4 <= n_nr && mr == MR {
                let cp = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                for t in 0u8..4 {
                    tile::preload_c(cp.add(t as usize * NR), n, t);
                }
                tile::microkernel_16x64_acc(
                    ap,
                    b_pack.as_ptr().add(jr * k * NR) as *const u8,
                    b_pack.as_ptr().add((jr + 1) * k * NR) as *const u8,
                    b_pack.as_ptr().add((jr + 2) * k * NR) as *const u8,
                    b_pack.as_ptr().add((jr + 3) * k * NR) as *const u8,
                    k,
                );
                for t in 0u8..4 {
                    tile::store_c(cp.add(t as usize * NR), n, t);
                }
                jr += 4;
            }

            // Double-wide: 2 B strips.
            while jr + 2 <= n_nr && mr == MR {
                let cp = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                tile::preload_c(cp, n, 0);
                tile::preload_c(cp.add(NR), n, 1);
                tile::microkernel_16x32_acc(
                    ap,
                    b_pack.as_ptr().add(jr * k * NR) as *const u8,
                    b_pack.as_ptr().add((jr + 1) * k * NR) as *const u8,
                    k,
                );
                tile::store_c(cp, n, 0);
                tile::store_c(cp.add(NR), n, 1);
                jr += 2;
            }

            // Single tiles.
            while jr < n_nr {
                let nr = NR.min(n - jr * NR);
                if mr == MR && nr == NR {
                    let cp = c.as_mut_ptr().add(ir * MR * n + jr * NR);
                    let bp = b_pack.as_ptr().add(jr * k * NR) as *const u8;
                    tile::preload_c(cp, n, 0);
                    tile::microkernel_16x16_acc(ap, bp, k);
                    tile::store_c(cp, n, 0);
                } else {
                    // Edge tile: scalar.
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
        }
    }

    drop(ctx);
}
