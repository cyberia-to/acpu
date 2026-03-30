//! AMX 16×16 f32 tile operations for GEMM microkernel.
//!
//! Computes C[16×16] += A[16×K] × B[K×16] using AMX fma32 outer
//! products. A is stored column-major (each column loaded into Y),
//! B is stored row-major (each row loaded into X).
//!
//! The microkernel batches 8 rank-1 updates at a time because
//! AMX has 8 X and 8 Y registers.

use super::asm::{amx_op, OP_FMA32, OP_LDX, OP_LDY, OP_STZ};
use super::fma::{fma_acc, fma_first};
use super::regs::{XRow, YRow};

/// AMX 16×16 f32 microkernel.
///
/// Computes `c[16×16] = a_panel × b_panel` (first call) or
/// `c[16×16] += a_panel × b_panel` (subsequent).
///
/// # Layout
///
/// - `b_panel`: K rows of 16 f32 each = K×64 bytes, row-major.
///   Each row is loaded into an X register.
/// - `a_panel`: K columns of 16 f32 each = K×64 bytes, column-major
///   (transposed). Each column is loaded into a Y register.
/// - Result accumulates in Z tile 0 (Z rows 0,4,8,...,60).
///
/// # Safety
///
/// - AMX must be active (caller holds AmxCtx).
/// - `a_panel` must have at least `k * 64` readable bytes, 64-byte aligned.
/// - `b_panel` must have at least `k * 64` readable bytes, 64-byte aligned.
/// - `k > 0`.
#[inline]
pub unsafe fn microkernel_16x16(a_panel: *const u8, b_panel: *const u8, k: usize) {
    let mut first = true;

    let mut p = 0usize;
    // Process 8 rank-1 updates at a time (fills all 8 X/Y registers).
    while p + 8 <= k {
        // Load 8 rows of B into X[0..7] and 8 columns of A into Y[0..7].
        for i in 0u8..8 {
            let b_ptr = b_panel.add((p + i as usize) * 64);
            let a_ptr = a_panel.add((p + i as usize) * 64);
            amx_op::<OP_LDX>((b_ptr as u64) | ((i as u64) << 56));
            amx_op::<OP_LDY>((a_ptr as u64) | ((i as u64) << 56));
        }

        // Issue 8 fma32 outer products.
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

    // Handle remaining 1..7 rank-1 updates.
    if p < k {
        let rem = k - p;
        for i in 0..(rem as u8) {
            let b_ptr = b_panel.add((p + i as usize) * 64);
            let a_ptr = a_panel.add((p + i as usize) * 64);
            amx_op::<OP_LDX>((b_ptr as u64) | ((i as u64) << 56));
            amx_op::<OP_LDY>((a_ptr as u64) | ((i as u64) << 56));
        }

        if first {
            amx_op::<OP_FMA32>(fma_first(XRow::new_unchecked(0), YRow::new_unchecked(0), 0));
            first = false;
            for i in 1..(rem as u8) {
                amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
            }
        } else {
            for i in 0..(rem as u8) {
                amx_op::<OP_FMA32>(fma_acc(XRow::new_unchecked(i), YRow::new_unchecked(i), 0));
            }
        }
    }
    let _ = first;
}

/// Store the 16×16 result from Z tile 0 into a row-major f32 buffer.
///
/// # Safety
///
/// - AMX must be active.
/// - `dst` must point to at least 16×64 = 1024 writable bytes, 64-byte aligned.
#[inline]
pub unsafe fn store_tile_16x16(dst: *mut u8) {
    // Z tile 0 uses rows 0, 4, 8, ..., 60.
    // Row j of the 16×16 result is in Z row j*4.
    for j in 0u8..16 {
        let z_row = j * 4; // Z rows: 0, 4, 8, ..., 60
        amx_op::<OP_STZ>((dst.add(j as usize * 64) as u64) | ((z_row as u64) << 56));
    }
}

/// Add the 16×16 result from Z tile 0 into an existing row-major f32 buffer.
///
/// Loads existing C values, adds Z tile contents, stores back.
///
/// # Safety
///
/// - AMX must be active.
/// - `dst` must point to at least 16×64 = 1024 readable+writable bytes.
#[inline]
pub unsafe fn accumulate_tile_16x16(c: *mut f32, ldc: usize) {
    // For each row j of the 16×16 tile, stored in Z row j*4:
    // Read Z row into a temp, add to C row, write back.
    // We use a stack buffer to extract Z rows.
    let mut z_row_buf = [0f32; 16];
    let z_ptr = z_row_buf.as_mut_ptr() as *mut u8;

    for j in 0u8..16 {
        let z_row = j * 4;
        amx_op::<OP_STZ>((z_ptr as u64) | ((z_row as u64) << 56));

        let c_row = c.add(j as usize * ldc);
        for (i, &zv) in z_row_buf.iter().enumerate() {
            *c_row.add(i) += zv;
        }
    }
}
