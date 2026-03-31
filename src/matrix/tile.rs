//! AMX f32 tile operations for GEMM microkernels.
//!
//! Two microkernels:
//! - `microkernel_16x16`: single tile, Z tile 0
//! - `microkernel_16x32`: double-wide, Z tiles 0+1, 2× compute per A load

use super::asm::{amx_op, OP_FMA32, OP_LDX, OP_LDY, OP_LDZ, OP_STZ};
use super::fma::{fma_acc, fma_first};
use super::regs::{XRow, YRow};

// ---------------------------------------------------------------------------
// C preload / direct store — eliminates AMX→CPU sync overhead
// ---------------------------------------------------------------------------

/// Load C[16×16] into Z tile via LDZ. No CPU involvement in data path.
///
/// # Safety
/// AMX must be active. `c` must point to `16 * ldc` readable f32 elements.
#[inline]
pub unsafe fn preload_c(c: *const f32, ldc: usize, tile: u8) {
    for j in 0u8..16 {
        let z_row = j * 4 + tile;
        let c_addr = (c as *const u8).add(j as usize * ldc * 4);
        amx_op::<OP_LDZ>((c_addr as u64) | ((z_row as u64) << 56));
    }
}

/// Store Z tile directly to C[16×16] via STZ. CPU never reads the data.
///
/// # Safety
/// AMX must be active. `c` must point to `16 * ldc` writable f32 elements.
#[inline]
pub unsafe fn store_c(c: *mut f32, ldc: usize, tile: u8) {
    for j in 0u8..16 {
        let z_row = j * 4 + tile;
        let c_addr = (c as *mut u8).add(j as usize * ldc * 4);
        amx_op::<OP_STZ>((c_addr as u64) | ((z_row as u64) << 56));
    }
}

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
// 16×16 microkernel — accumulate-only (Z preloaded with C)
// ---------------------------------------------------------------------------

/// AMX 16×16 microkernel that accumulates into existing Z tile 0.
/// Use after `preload_c` to compute C += A × B without AMX→CPU sync.
///
/// # Safety
/// AMX must be active. Z tile must be preloaded. Panels: 64-byte aligned, `k*64` bytes.
#[inline]
pub unsafe fn microkernel_16x16_acc(a_panel: *const u8, b_panel: *const u8, k: usize, bs: usize) {
    let mut p = 0usize;

    while p + 8 <= k {
        if p + 16 <= k {
            for i in (0..8).step_by(4) {
                crate::sync::prefetch::prefetch_l1(b_panel.add((p + 8 + i) * bs));
                crate::sync::prefetch::prefetch_l1(a_panel.add((p + 8 + i) * 64));
            }
        }
        for i in 0u8..8 {
            amx_op::<OP_LDX>((b_panel.add((p + i as usize) * bs) as u64) | ((i as u64) << 56));
            amx_op::<OP_LDY>((a_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }
        for i in 0u8..8 {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
        }
        p += 8;
    }

    while p < k {
        amx_op::<OP_LDX>(b_panel.add(p * bs) as u64);
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
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
// Fused AMX batch: single asm block eliminates LLVM fences between ops
// ---------------------------------------------------------------------------

/// 18 AMX ops in ONE asm block: 2 LDY + 8 LDX + 8 FMA.
/// No LLVM-inserted fences between instructions.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_batch_16x64(
    a0: u64,
    a1: u64,
    bx0: u64,
    bx1: u64,
    bx2: u64,
    bx3: u64,
    bx4: u64,
    bx5: u64,
    bx6: u64,
    bx7: u64,
    f0: u64,
    f1: u64,
    f2: u64,
    f3: u64,
    g0: u64,
    g1: u64,
    g2: u64,
    g3: u64,
) {
    // Interleaved: overlap loads with FMA for AMX pipeline utilization.
    // op 1 = LDY, op 0 = LDX, op 12 = FMA32
    core::arch::asm!(
        // Load Y[0], start loading X[0..1]
        ".word (0x00201000 + (1 << 5) + 0{a0} - ((0{a0} >> 4) * 6))",
        ".word (0x00201000 + (0 << 5) + 0{bx0} - ((0{bx0} >> 4) * 6))",
        ".word (0x00201000 + (0 << 5) + 0{bx1} - ((0{bx1} >> 4) * 6))",
        // FMA tile 0 while X[2] loads
        ".word (0x00201000 + (0 << 5) + 0{bx2} - ((0{bx2} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{f0} - ((0{f0} >> 4) * 6))",
        // FMA tile 1 while X[3] loads
        ".word (0x00201000 + (0 << 5) + 0{bx3} - ((0{bx3} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{f1} - ((0{f1} >> 4) * 6))",
        // FMA tiles 2-3, load Y[1]
        ".word (0x00201000 + (12 << 5) + 0{f2} - ((0{f2} >> 4) * 6))",
        ".word (0x00201000 + (1 << 5) + 0{a1} - ((0{a1} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{f3} - ((0{f3} >> 4) * 6))",
        // Load X[4..5], FMA tile 0 with Y[1]
        ".word (0x00201000 + (0 << 5) + 0{bx4} - ((0{bx4} >> 4) * 6))",
        ".word (0x00201000 + (0 << 5) + 0{bx5} - ((0{bx5} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{g0} - ((0{g0} >> 4) * 6))",
        // Load X[6..7], FMA tiles 1-3 with Y[1]
        ".word (0x00201000 + (0 << 5) + 0{bx6} - ((0{bx6} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{g1} - ((0{g1} >> 4) * 6))",
        ".word (0x00201000 + (0 << 5) + 0{bx7} - ((0{bx7} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{g2} - ((0{g2} >> 4) * 6))",
        ".word (0x00201000 + (12 << 5) + 0{g3} - ((0{g3} >> 4) * 6))",
        a0 = in(reg) a0,
        a1 = in(reg) a1,
        bx0 = in(reg) bx0,
        bx1 = in(reg) bx1,
        bx2 = in(reg) bx2,
        bx3 = in(reg) bx3,
        bx4 = in(reg) bx4,
        bx5 = in(reg) bx5,
        bx6 = in(reg) bx6,
        bx7 = in(reg) bx7,
        f0 = in(reg) f0,
        f1 = in(reg) f1,
        f2 = in(reg) f2,
        f3 = in(reg) f3,
        g0 = in(reg) g0,
        g1 = in(reg) g1,
        g2 = in(reg) g2,
        g3 = in(reg) g3,
        options(nostack),
    );
}

// ---------------------------------------------------------------------------
// 16×64 microkernel (quad-wide: all 4 Z tiles in N direction)
// ---------------------------------------------------------------------------

/// AMX 16×64 f32 microkernel. Uses all 4 Z tiles.
///
/// Each Y load (A column) serves 4 FMAs — maximum register reuse.
/// Batch of 2: 2Y + 8X + 8 FMA = 18 instructions for 4096 FLOPS.
///
/// # Safety
/// AMX must be active. All panels must be 64-byte aligned with `k*64` readable bytes.
#[inline]
pub unsafe fn microkernel_16x64(
    a_panel: *const u8,
    b0: *const u8,
    b1: *const u8,
    b2: *const u8,
    b3: *const u8,
    k: usize,
) {
    let mut p = 0usize;

    // First batch: use skip_z (bit 27) to clear all 4 tiles.
    if p + 2 <= k {
        let a0 = a_panel.add(p * 64) as u64;
        let a1 = (a_panel.add((p + 1) * 64) as u64) | (1u64 << 56);
        let bx0 = b0.add(p * 64) as u64;
        let bx1 = (b1.add(p * 64) as u64) | (1u64 << 56);
        let bx2 = (b2.add(p * 64) as u64) | (2u64 << 56);
        let bx3 = (b3.add(p * 64) as u64) | (3u64 << 56);
        let bx4 = (b0.add((p + 1) * 64) as u64) | (4u64 << 56);
        let bx5 = (b1.add((p + 1) * 64) as u64) | (5u64 << 56);
        let bx6 = (b2.add((p + 1) * 64) as u64) | (6u64 << 56);
        let bx7 = (b3.add((p + 1) * 64) as u64) | (7u64 << 56);

        // fma_first operands (skip_z=1) for tiles 0-3.
        let f0 = fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
        let f1 = fma_first(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
        let f2 = fma_first(XRow::new_unchecked(2), YRow::new_unchecked(0), 2);
        let f3 = fma_first(XRow::new_unchecked(3), YRow::new_unchecked(0), 3);
        let g0 = fma_acc(XRow::new_unchecked(4), YRow::new_unchecked(1), 0);
        let g1 = fma_acc(XRow::new_unchecked(5), YRow::new_unchecked(1), 1);
        let g2 = fma_acc(XRow::new_unchecked(6), YRow::new_unchecked(1), 2);
        let g3 = fma_acc(XRow::new_unchecked(7), YRow::new_unchecked(1), 3);

        // Single asm block: 18 AMX ops, no LLVM fences between them.
        fused_batch_16x64(
            a0, a1, bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7, f0, f1, f2, f3, g0, g1, g2, g3,
        );
        p += 2;
    }

    // Steady-state: accumulate (no skip_z).
    while p + 2 <= k {
        if p + 4 <= k {
            crate::sync::prefetch::prefetch_l1(a_panel.add((p + 2) * 64));
            crate::sync::prefetch::prefetch_l1(b0.add((p + 2) * 64));
        }

        let a0 = a_panel.add(p * 64) as u64;
        let a1 = (a_panel.add((p + 1) * 64) as u64) | (1u64 << 56);
        let bx0 = b0.add(p * 64) as u64;
        let bx1 = (b1.add(p * 64) as u64) | (1u64 << 56);
        let bx2 = (b2.add(p * 64) as u64) | (2u64 << 56);
        let bx3 = (b3.add(p * 64) as u64) | (3u64 << 56);
        let bx4 = (b0.add((p + 1) * 64) as u64) | (4u64 << 56);
        let bx5 = (b1.add((p + 1) * 64) as u64) | (5u64 << 56);
        let bx6 = (b2.add((p + 1) * 64) as u64) | (6u64 << 56);
        let bx7 = (b3.add((p + 1) * 64) as u64) | (7u64 << 56);

        let f0 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
        let f1 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
        let f2 = fma_acc(XRow::new_unchecked(2), YRow::new_unchecked(0), 2);
        let f3 = fma_acc(XRow::new_unchecked(3), YRow::new_unchecked(0), 3);
        let g0 = fma_acc(XRow::new_unchecked(4), YRow::new_unchecked(1), 0);
        let g1 = fma_acc(XRow::new_unchecked(5), YRow::new_unchecked(1), 1);
        let g2 = fma_acc(XRow::new_unchecked(6), YRow::new_unchecked(1), 2);
        let g3 = fma_acc(XRow::new_unchecked(7), YRow::new_unchecked(1), 3);

        fused_batch_16x64(
            a0, a1, bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7, f0, f1, f2, f3, g0, g1, g2, g3,
        );
        p += 2;
    }

    // Remainder.
    if p < k {
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_LDX>(b0.add(p * 64) as u64);
        amx_op::<OP_LDX>((b1.add(p * 64) as u64) | (1u64 << 56));
        amx_op::<OP_LDX>((b2.add(p * 64) as u64) | (2u64 << 56));
        amx_op::<OP_LDX>((b3.add(p * 64) as u64) | (3u64 << 56));

        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(2), YRow::new_unchecked(0), 2));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(3), YRow::new_unchecked(0), 3));
    }
}

// ---------------------------------------------------------------------------
// 16×64 microkernel — accumulate-only (Z preloaded with C)
// ---------------------------------------------------------------------------

/// AMX 16×64 microkernel that accumulates into existing Z tiles 0-3.
/// `bs` = B stride in bytes (64 for packed NR-wide strips, n*4 for direct row-major).
///
/// # Safety
/// AMX must be active. Z tiles must be preloaded. A panel: 64-byte stride.
/// B panels: `bs`-byte stride, each row = 64 readable bytes.
#[inline]
pub unsafe fn microkernel_16x64_acc(
    a_panel: *const u8,
    b0: *const u8,
    b1: *const u8,
    b2: *const u8,
    b3: *const u8,
    k: usize,
    bs: usize,
) {
    let mut p = 0usize;

    while p + 2 <= k {
        if p + 4 <= k {
            crate::sync::prefetch::prefetch_l1(a_panel.add((p + 2) * 64));
            crate::sync::prefetch::prefetch_l1(b0.add((p + 2) * bs));
        }

        let a0 = a_panel.add(p * 64) as u64;
        let a1 = (a_panel.add((p + 1) * 64) as u64) | (1u64 << 56);
        let bx0 = b0.add(p * bs) as u64;
        let bx1 = (b1.add(p * bs) as u64) | (1u64 << 56);
        let bx2 = (b2.add(p * bs) as u64) | (2u64 << 56);
        let bx3 = (b3.add(p * bs) as u64) | (3u64 << 56);
        let bx4 = (b0.add((p + 1) * bs) as u64) | (4u64 << 56);
        let bx5 = (b1.add((p + 1) * bs) as u64) | (5u64 << 56);
        let bx6 = (b2.add((p + 1) * bs) as u64) | (6u64 << 56);
        let bx7 = (b3.add((p + 1) * bs) as u64) | (7u64 << 56);

        let f0 = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
        let f1 = fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1);
        let f2 = fma_acc(XRow::new_unchecked(2), YRow::new_unchecked(0), 2);
        let f3 = fma_acc(XRow::new_unchecked(3), YRow::new_unchecked(0), 3);
        let g0 = fma_acc(XRow::new_unchecked(4), YRow::new_unchecked(1), 0);
        let g1 = fma_acc(XRow::new_unchecked(5), YRow::new_unchecked(1), 1);
        let g2 = fma_acc(XRow::new_unchecked(6), YRow::new_unchecked(1), 2);
        let g3 = fma_acc(XRow::new_unchecked(7), YRow::new_unchecked(1), 3);

        fused_batch_16x64(
            a0, a1, bx0, bx1, bx2, bx3, bx4, bx5, bx6, bx7, f0, f1, f2, f3, g0, g1, g2, g3,
        );
        p += 2;
    }

    if p < k {
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_LDX>(b0.add(p * bs) as u64);
        amx_op::<OP_LDX>((b1.add(p * bs) as u64) | (1u64 << 56));
        amx_op::<OP_LDX>((b2.add(p * bs) as u64) | (2u64 << 56));
        amx_op::<OP_LDX>((b3.add(p * bs) as u64) | (3u64 << 56));

        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(2), YRow::new_unchecked(0), 2));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(3), YRow::new_unchecked(0), 3));
    }
}

// ---------------------------------------------------------------------------
// 16×32 microkernel — accumulate-only (Z preloaded with C)
// ---------------------------------------------------------------------------

/// AMX 16×32 microkernel that accumulates into existing Z tiles 0-1.
/// Use after `preload_c` for both tiles.
///
/// # Safety
/// AMX must be active. Z tiles must be preloaded. All panels: 64-byte aligned, `k*64` bytes.
#[inline]
pub unsafe fn microkernel_16x32_acc(
    a_panel: *const u8,
    b_left: *const u8,
    b_right: *const u8,
    k: usize,
    bs: usize,
) {
    let mut p = 0usize;

    while p + 4 <= k {
        if p + 8 <= k {
            crate::sync::prefetch::prefetch_l1(a_panel.add((p + 4) * 64));
            crate::sync::prefetch::prefetch_l1(b_left.add((p + 4) * bs));
            crate::sync::prefetch::prefetch_l1(b_right.add((p + 4) * bs));
        }

        for i in 0u8..4 {
            amx_op::<OP_LDY>((a_panel.add((p + i as usize) * 64) as u64) | ((i as u64) << 56));
        }
        for i in 0u8..4 {
            amx_op::<OP_LDX>((b_left.add((p + i as usize) * bs) as u64) | ((i as u64) << 56));
        }
        for i in 0u8..4 {
            amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
        }

        for i in 0u8..4 {
            amx_op::<OP_LDX>(
                (b_right.add((p + i as usize) * bs) as u64) | (((4 + i) as u64) << 56),
            );
        }
        for i in 0u8..4 {
            amx_op::<OP_FMA32>(fma_acc(
                XRow::new_unchecked(4 + i),
                YRow::new_unchecked(i),
                1,
            ));
        }

        p += 4;
    }

    while p < k {
        amx_op::<OP_LDY>(a_panel.add(p * 64) as u64);
        amx_op::<OP_LDX>(b_left.add(p * bs) as u64);
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));

        amx_op::<OP_LDX>((b_right.add(p * bs) as u64) | (1u64 << 56));
        amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(1), YRow::new_unchecked(0), 1));

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
