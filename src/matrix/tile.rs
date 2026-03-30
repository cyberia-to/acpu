//! AMX f32 tile operations for GEMM microkernels.
//!
//! Two microkernels:
//! - `microkernel_16x16`: single tile, Z tile 0
//! - `microkernel_16x32`: double-wide, Z tiles 0+1, 2× compute per A load

use super::asm::{amx_op, OP_FMA32, OP_LDX, OP_LDY, OP_STZ};
use super::fma::{fma_acc, fma_first};
use super::regs::{XRow, YRow};

// ---------------------------------------------------------------------------
// 16×16 microkernel (single tile)
// ---------------------------------------------------------------------------

/// AMX 16×16 f32 microkernel. Z tile 0.
///
/// # Safety
/// AMX must be active. Panels must be 64-byte aligned with `k*64` readable bytes.
#[inline]
pub unsafe fn microkernel_16x16(a_panel: *const u8, b_panel: *const u8, k: usize) {
    let mut first = true;
    let mut p = 0usize;

    while p + 8 <= k {
        // Prefetch next batch while loading current.
        if p + 16 <= k {
            for i in (0..8).step_by(4) {
                crate::sync::prefetch::prefetch_l1(b_panel.add((p + 8 + i) * 64));
                crate::sync::prefetch::prefetch_l1(a_panel.add((p + 8 + i) * 64));
            }
        }
        for i in 0u8..8 {
            amx_op::<OP_LDX>((b_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
            amx_op::<OP_LDY>((a_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }
        if first {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
            first = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        }
        for i in 1u8..8 {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
        }
        p += 8;
    }

    while p < k {
        amx_op::<OP_LDX>(b_panel.add(p * 64) as u64);
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        if first {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
            first = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        }
        p += 1;
    }
}

// ---------------------------------------------------------------------------
// 16×32 microkernel (double-wide: 2 tiles side by side in N)
// ---------------------------------------------------------------------------

/// AMX 16×32 f32 microkernel. Uses Z tiles 0 and 1.
///
/// Computes C[16×32] += A[16×K] × B[K×32] using 2 tiles:
/// - Tile 0: C[0..16, 0..16]   = A × B_left
/// - Tile 1: C[0..16, 16..32]  = A × B_right
///
/// # Safety
/// AMX must be active. All panels must be 64-byte aligned with `k*64` readable bytes.
#[inline]
pub unsafe fn microkernel_16x32(
    a_panel: *const u8,
    b_left: *const u8,
    b_right: *const u8,
    k: usize,
) {
    let mut first_t0 = true;
    let mut first_t1 = true;
    let mut p = 0usize;

    // Batch of 4: 4Y + 4X(left) + 4X(right) = 4Y + 8X = all registers used.
    while p + 4 <= k {
        // Prefetch next batch.
        if p + 8 <= k {
            crate::sync::prefetch::prefetch_l1(a_panel.add((p + 4) * 64));
            crate::sync::prefetch::prefetch_l1(b_left.add((p + 4) * 64));
            crate::sync::prefetch::prefetch_l1(b_right.add((p + 4) * 64));
        }

        for i in 0u8..4 {
            amx_op::<OP_LDY>((a_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }
        for i in 0u8..4 {
            amx_op::<OP_LDX>((b_left.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }

        // FMA tile 0
        if first_t0 {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
            first_t0 = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        }
        for i in 1u8..4 {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
        }

        // Load B_right into X[4..7]
        for i in 0u8..4 {
            amx_op::<OP_LDX>(
                (b_right.add((p + i as usize) * 64) as u64) | (((4 + i) as u64) << 56),
            );
        }

        // FMA tile 1 (reuses Y[0..3])
        if first_t1 {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(4), YRow::new_unchecked(0), 1));
            first_t1 = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(4), YRow::new_unchecked(0), 1));
        }
        for i in 1u8..4 {
            amx_op::<OP_FMA32>(fma_acc(
                XRow::new_unchecked(4 + i),
                YRow::new_unchecked(i),
                1,
            ));
        }

        p += 4;
    }

    // Remainder: one at a time.
    while p < k {
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_LDX>(b_left.add(p * 64) as u64);

        if first_t0 {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
            first_t0 = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        }

        amx_op::<OP_LDX>((b_right.add(p * 64) as u64) | (1u64 << 56));
        if first_t1 {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(1), YRow::new_unchecked(0), 1));
            first_t1 = false;
        } else {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1));
        }

        p += 1;
    }
}

// ---------------------------------------------------------------------------
// Tile store / accumulate
// ---------------------------------------------------------------------------

/// Store the 16×16 result from Z tile 0 into a row-major f32 buffer.
///
/// # Safety
/// AMX must be active. `dst` must have 1024 writable bytes, 64-byte aligned.
#[inline]
pub unsafe fn store_tile_16x16(dst: *mut u8) {
    for j in 0u8..16 {
        let z_row = j * 4;
        amx_op::<OP_STZ>((dst.add(j as usize * 64) as u64) | ((z_row as u64) << 56));
    }
}

/// Add Z tile contents into existing C buffer. NEON-vectorized.
///
/// # Safety
/// AMX must be active. `c` must point to valid f32 data with stride `ldc`.
#[inline]
pub unsafe fn accumulate_tile(c: *mut f32, ldc: usize, tile: u8) {
    #[repr(align(64))]
    struct A64([f32; 16]);

    let mut zbuf = A64([0f32; 16]);
    let z_ptr = zbuf.0.as_mut_ptr() as *mut u8;

    for j in 0u8..16 {
        let z_row = j * 4 + tile;
        amx_op::<OP_STZ>((z_ptr as u64) | ((z_row as u64) << 56));

        let c_row = c.add(j as usize * ldc);

        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32};
            for q in 0..4usize {
                let existing = vld1q_f32(c_row.add(q * 4));
                let z_val = vld1q_f32(zbuf.0.as_ptr().add(q * 4));
                vst1q_f32(c_row.add(q * 4), vaddq_f32(existing, z_val));
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..16 {
                *c_row.add(i) += zbuf.0[i];
            }
        }
    }
}

/// Backward-compatible wrapper for tile 0.
///
/// # Safety
/// AMX must be active. `c` must point to valid f32 data with stride `ldc`.
#[inline]
pub unsafe fn accumulate_tile_16x16(c: *mut f32, ldc: usize) {
    accumulate_tile(c, ldc, 0);
}
