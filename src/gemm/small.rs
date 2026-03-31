//! Direct AMX path for small matrices. Zero heap allocation.
//! No B packing — loads B directly with row stride via LDX.
//! A packed strip-by-strip (hot in L1 when used). Preload C → compute → store.

use super::{MR, NR};
use crate::matrix::tile;

/// Direct AMX sgemm for small matrices where n*k ≤ 32K.
/// No B packing, NEON A packing per-strip, thread-local warm buffer.
///
/// Per-strip packing: pack strip 0, compute strip 0 (AMX), pack strip 1 (NEON),
/// compute strip 1, ... The NEON pack after each strip's compute overlaps with
/// the AMX pipeline draining — AMX FMA and NEON ALU use separate execution units.
#[cfg(target_arch = "aarch64")]
pub(super) fn sgemm_amx_direct(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    super::ensure_amx();

    let n_mr = m.div_ceil(MR);
    let a_need = n_mr * MR * k;
    let mut a_pack = super::PACK_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        super::cached_buf(&mut cache.a, a_need)
    });

    let bs = n * 4; // B stride in bytes
    let n_nr = n.div_ceil(NR);

    // Pack first strip (must be ready before first compute).
    let first_mc = MR.min(m);
    super::pack_a_mr(a, k, 0, 0, first_mc, k, a_pack.as_mut_slice());

    unsafe {
        for ir in 0..n_mr {
            let mr = MR.min(m - ir * MR);
            let ap = a_pack.as_slice().as_ptr().add(ir * k * MR) as *const u8;

            // ---- AMX compute for strip ir ----
            let mut jr = 0usize;

            // Quad-wide: 4 B strips, direct from b[].
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

            // Double-wide.
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

            // Single tiles.
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

            // ---- Pack next strip (NEON, overlaps with AMX pipeline drain) ----
            // AMX FMA/STZ ops are still in-flight; NEON pack uses ALU+load
            // ports that don't compete with AMX execution units.
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

    // Return buffer to thread-local cache for reuse.
    super::PACK_CACHE.with(|c| {
        c.borrow_mut().a = Some(a_pack);
    });
}
