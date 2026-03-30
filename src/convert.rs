//! Convenience re-exports for numeric type conversions.
//!
//! All conversion functions are available directly under `ramx::convert::*`
//! or via the top-level `ramx::cvt_*` aliases.

pub use crate::numeric::bf16::{bf16_to_f32, cvt_bf16_f32, cvt_f32_bf16, f32_to_bf16};
pub use crate::numeric::fp16::{cvt_f16_f32, cvt_f32_f16, f32_to_fp16, fp16_to_f32};
pub use crate::numeric::quant::{cvt_f32_i8, cvt_i8_f32};
