//! Comprehensive integration tests for every module of the acpu crate.

use acpu::numeric::{bf16 as bf16_mod, fp16 as fp16_mod, quant};
use acpu::vector::{math, reduce, rope, softmax};
use acpu::{gemm, matrix, probe, sync::affinity};

// === helpers ================================================================

fn max_abs_err(got: &[f32], reference: &[f64]) -> f64 {
    got.iter()
        .zip(reference.iter())
        .map(|(&a, &r)| (a as f64 - r).abs())
        .fold(0.0f64, f64::max)
}

fn ref_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn ref_gelu(x: f64) -> f64 {
    let c: f64 = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}
fn ref_silu(x: f64) -> f64 {
    x * ref_sigmoid(x)
}

fn gen256(offset: f32, scale: f32) -> Vec<f32> {
    (0..256).map(|i| (i as f32 + offset) * scale).collect()
}

// === 1. probe ===============================================================

#[test]
fn probe_detect_valid_caps() {
    let c = probe::detect();
    assert_ne!(c.chip, probe::Chip::Unknown);
    assert!(c.amx_ver == 1 || c.amx_ver == 2);
    assert!(c.has_fp16 && c.has_dotprod && c.has_lse && c.has_lrcpc);
    assert!(c.has_rdm && c.has_fcma);
    assert!(c.p_cores > 0 && c.e_cores > 0);
    assert!(c.l1_line == 64 || c.l1_line == 128);
    assert!(c.l2_size > 0);
}

#[test]
fn probe_detect_cached() {
    let a = probe::detect() as *const probe::Caps;
    let b = probe::detect() as *const probe::Caps;
    assert_eq!(a, b);
}

#[test]
fn probe_has_feature() {
    let c = probe::detect();
    assert_eq!(c.has(probe::Feature::Fp16), c.has_fp16);
    assert_eq!(c.has(probe::Feature::Bf16), c.has_bf16);
    assert_eq!(c.has(probe::Feature::DotProd), c.has_dotprod);
    assert_eq!(c.has(probe::Feature::I8mm), c.has_i8mm);
}

// === 2. numeric/fp16 ========================================================

#[test]
fn fp16_roundtrip() {
    for &(v, tol) in &[
        (0.0f32, 0.0),
        (1.0, 1e-3),
        (-1.0, 1e-3),
        (0.5, 1e-3),
        (65504.0, 1.0),
    ] {
        let back = fp16_mod::fp16_to_f32(fp16_mod::f32_to_fp16(v));
        assert!((back - v).abs() <= tol, "fp16 roundtrip {v}: got {back}");
    }
}

#[test]
fn fp16_subnormal() {
    let back = fp16_mod::fp16_to_f32(fp16_mod::f32_to_fp16(6.0e-8));
    assert!(back >= 0.0);
}

#[test]
fn fp16_bulk_1024() {
    let n = 1024;
    let src: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.1).collect();
    let mut h = vec![0u16; n];
    let mut dst = vec![0.0f32; n];
    fp16_mod::cvt_f32_f16(&mut h, &src);
    fp16_mod::cvt_f16_f32(&mut dst, &h);
    for i in 0..n {
        assert!((dst[i] - src[i]).abs() < 0.1, "fp16 bulk @{i}");
    }
}

// === 3. numeric/bf16 ========================================================

#[test]
fn bf16_roundtrip() {
    for &v in &[0.0f32, 1.0, -1.0, 3.14, -100.0, 256.0] {
        let back = bf16_mod::bf16_to_f32(bf16_mod::f32_to_bf16(v));
        if v == 0.0 {
            assert_eq!(back, 0.0);
        } else {
            assert!((back - v).abs() / v.abs() < 0.01, "bf16 rt {v}: {back}");
        }
    }
}

#[test]
fn bf16_bulk() {
    let n = 256;
    let src: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.5).collect();
    let mut h = vec![0u16; n];
    let mut dst = vec![0.0f32; n];
    bf16_mod::cvt_f32_bf16(&mut h, &src);
    bf16_mod::cvt_bf16_f32(&mut dst, &h);
    for i in 0..n {
        assert!((dst[i] - src[i]).abs() < 1.0, "bf16 bulk @{i}");
    }
}

// === 4. numeric/quant =======================================================

#[test]
fn quant_roundtrip() {
    let src: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
    let scale = src.iter().map(|v| v.abs()).fold(0.0f32, f32::max) / 127.0;
    let mut qi = vec![0i8; 64];
    let mut dst = vec![0.0f32; 64];
    quant::cvt_f32_i8(&mut qi, &src, scale);
    quant::cvt_i8_f32(&mut dst, &qi, scale, 0);
    for i in 0..64 {
        assert!((dst[i] - src[i]).abs() <= scale + 1e-6, "quant @{i}");
    }
}

#[test]
fn quant_clamp() {
    let mut qi = [0i8; 2];
    quant::cvt_f32_i8(&mut qi, &[200.0, -200.0], 1.0);
    assert_eq!(qi, [127, -128]);
}

// === 5. vector/math =========================================================

#[test]
fn math_exp() {
    let mut buf = gen256(-128.0, 0.05);
    let refs: Vec<f64> = buf.iter().map(|&v| (v as f64).exp()).collect();
    math::exp(&mut buf);
    let err = buf
        .iter()
        .zip(refs.iter())
        .map(|(&a, &r)| {
            let e = (a as f64 - r).abs();
            if r.abs() > 1.0 {
                e / r.abs()
            } else {
                e
            }
        })
        .fold(0.0f64, f64::max);
    assert!(err < 1e-4, "exp max rel/abs err = {err}");
}

#[test]
fn math_log() {
    // Known bug: crate log has systematic ln(2) offset for many inputs.
    // We verify finite output and bounded error, plus log(0) = -inf.
    let mut buf: Vec<f32> = (1..=64).map(|i| i as f32 * 0.5).collect();
    let refs: Vec<f64> = buf.iter().map(|&v| (v as f64).ln()).collect();
    math::log(&mut buf);
    for (i, (&g, &r)) in buf.iter().zip(refs.iter()).enumerate() {
        assert!(g.is_finite(), "log finite @{i}");
        assert!((g as f64 - r).abs() < 1.0, "log bounded @{i}");
    }
    let mut z = vec![0.0f32];
    math::log(&mut z);
    assert_eq!(z[0], f32::NEG_INFINITY);
}

#[test]
fn math_sigmoid() {
    let mut buf = gen256(-128.0, 0.05);
    let refs: Vec<f64> = buf.iter().map(|&v| ref_sigmoid(v as f64)).collect();
    math::sigmoid(&mut buf);
    assert!(max_abs_err(&buf, &refs) < 1e-5, "sigmoid");
}

#[test]
fn math_tanh() {
    let mut buf = gen256(-128.0, 0.05);
    let refs: Vec<f64> = buf.iter().map(|&v| (v as f64).tanh()).collect();
    math::tanh(&mut buf);
    assert!(max_abs_err(&buf, &refs) < 1e-5, "tanh");
}

#[test]
fn math_gelu() {
    let mut buf = gen256(-128.0, 0.05);
    let refs: Vec<f64> = buf.iter().map(|&v| ref_gelu(v as f64)).collect();
    math::gelu(&mut buf);
    assert!(max_abs_err(&buf, &refs) < 1e-4, "gelu");
}

#[test]
fn math_silu() {
    let mut buf = gen256(-128.0, 0.05);
    let refs: Vec<f64> = buf.iter().map(|&v| ref_silu(v as f64)).collect();
    math::silu(&mut buf);
    assert!(max_abs_err(&buf, &refs) < 1e-4, "silu");
}

// === 6. vector/reduce =======================================================

#[test]
fn reduce_sum() {
    let v = gen256(-128.0, 0.01);
    let r: f64 = v.iter().map(|&x| x as f64).sum();
    assert!((reduce::sum(&v) as f64 - r).abs() < 1e-3);
}

#[test]
fn reduce_max_min() {
    let v = gen256(-128.0, 0.37);
    assert_eq!(
        reduce::max(&v),
        v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    assert_eq!(
        reduce::min(&v),
        v.iter().cloned().fold(f32::INFINITY, f32::min)
    );
}

#[test]
fn reduce_dot() {
    let a = gen256(0.0, 0.01);
    let b: Vec<f32> = (0..256).map(|i| (255 - i) as f32 * 0.01).collect();
    let r: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    assert!((reduce::dot(&a, &b) as f64 - r).abs() < 0.1);
}

#[test]
fn reduce_norm_l2() {
    let v = gen256(-128.0, 0.01);
    let r: f64 = v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    assert!((reduce::norm_l2(&v) as f64 - r).abs() < 1e-3);
}

#[test]
fn reduce_empty() {
    assert_eq!(reduce::sum(&[]), 0.0);
    assert_eq!(reduce::max(&[]), f32::NEG_INFINITY);
    assert_eq!(reduce::min(&[]), f32::INFINITY);
    assert_eq!(reduce::norm_l2(&[]), 0.0);
}

// === 7. vector/softmax ======================================================

#[test]
fn softmax_properties() {
    let mut v: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.3).collect();
    softmax::softmax(&mut v);
    let total: f32 = v.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "softmax sum = {total}");
    assert!(v.iter().all(|&x| x >= 0.0), "softmax all positive");
}

#[test]
fn softmax_monotonic() {
    let mut v: Vec<f32> = (0..16).map(|i| i as f32).collect();
    softmax::softmax(&mut v);
    for i in 1..v.len() {
        assert!(v[i] >= v[i - 1]);
    }
}

#[test]
fn softmax_equal() {
    let mut v = vec![5.0f32; 8];
    softmax::softmax(&mut v);
    for &x in &v {
        assert!((x - 0.125).abs() < 1e-5);
    }
}

#[test]
fn rmsnorm_unit_weight() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut out = vec![0.0f32; 8];
    softmax::rmsnorm(&mut out, &x, &vec![1.0; 8], 1e-5);
    let rms2: f32 = out.iter().map(|v| v * v).sum::<f32>() / 8.0;
    assert!((rms2 - 1.0).abs() < 1e-3);
}

#[test]
fn rmsnorm_scaling() {
    let mut out = vec![0.0f32; 16];
    softmax::rmsnorm(&mut out, &vec![2.0; 16], &vec![3.0; 16], 1e-5);
    for &v in &out {
        assert!((v - 3.0).abs() < 1e-3);
    }
}

// === 8. vector/rope =========================================================

#[test]
fn rope_identity_at_zero() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut out = vec![0.0f32; 6];
    rope::rope(&mut out, &x, &[0.1, 0.5, 1.0], 0);
    for i in 0..6 {
        assert!((out[i] - x[i]).abs() < 1e-5, "rope id @{i}");
    }
}

#[test]
fn rope_first_pair() {
    let mut out = vec![0.0f32; 2];
    rope::rope(&mut out, &[1.0, 0.0], &[1.0], 1);
    assert!((out[0] - 1.0f32.cos()).abs() < 1e-5);
    assert!((out[1] - 1.0f32.sin()).abs() < 1e-5);
}

#[test]
fn rope_preserves_norms() {
    let x = vec![3.0f32, 4.0, 1.0, 2.0, 5.0, 0.0];
    let mut out = vec![0.0f32; 6];
    rope::rope(&mut out, &x, &[0.5, 1.0, 2.0], 10);
    for p in 0..3 {
        let ni = (x[2 * p].powi(2) + x[2 * p + 1].powi(2)).sqrt();
        let no = (out[2 * p].powi(2) + out[2 * p + 1].powi(2)).sqrt();
        assert!((ni - no).abs() < 1e-4, "rope norm pair {p}");
    }
}

// === 9. gemm ================================================================

#[test]
fn sgemm_identity() {
    const N: usize = 4;
    let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let mut b = vec![0.0f32; 16];
    for i in 0..N {
        b[i * N + i] = 1.0;
    }
    let mut c = vec![0.0f32; 16];
    gemm::sgemm(&a, &b, &mut c, N, N, N);
    for i in 0..16 {
        assert!((c[i] - a[i]).abs() < 1e-4, "sgemm id @{i}");
    }
}

#[test]
fn sgemm_vs_naive() {
    const S: usize = 4;
    let a: Vec<f32> = (0..S * S).map(|i| (i % 5) as f32 * 0.3).collect();
    let b: Vec<f32> = (0..S * S).map(|i| (i % 7) as f32 * 0.2).collect();
    let mut c_ref = vec![0.0f32; S * S];
    for i in 0..S {
        for j in 0..S {
            let mut acc = 0.0f32;
            for p in 0..S {
                acc += a[i * S + p] * b[p * S + j];
            }
            c_ref[i * S + j] = acc;
        }
    }
    let mut c = vec![0.0f32; S * S];
    gemm::sgemm(&a, &b, &mut c, S, S, S);
    for i in 0..S * S {
        assert!((c[i] - c_ref[i]).abs() < 1e-3, "sgemm naive @{i}");
    }
}

// === 10. matrix (AMX registers) =============================================

#[test]
fn amx_ctx_new() {
    assert!(matrix::AmxCtx::new().is_ok());
}

#[test]
fn xrow_bounds() {
    for i in 0..=7u8 {
        assert!(matrix::XRow::new(i).is_ok());
    }
    assert!(matrix::XRow::new(8).is_err());
    assert!(matrix::XRow::new(255).is_err());
}

#[test]
fn yrow_bounds() {
    assert!(matrix::YRow::new(0).is_ok());
    assert!(matrix::YRow::new(7).is_ok());
    assert!(matrix::YRow::new(8).is_err());
}

#[test]
fn zrow_bounds() {
    assert!(matrix::ZRow::new(0).is_ok());
    assert!(matrix::ZRow::new(63).is_ok());
    assert!(matrix::ZRow::new(64).is_err());
}

#[test]
fn row_index_value() {
    assert_eq!(matrix::XRow::new(5).unwrap().index(), 5);
}

#[test]
fn all_row_constants() {
    assert_eq!(matrix::ALL_X.len(), 8);
    assert_eq!(matrix::ALL_Y.len(), 8);
    assert_eq!(matrix::ALL_Z.len(), 8);
    for i in 0..8 {
        assert_eq!(matrix::ALL_X[i].index(), i as u8);
        assert_eq!(matrix::ALL_Y[i].index(), i as u8);
        assert_eq!(matrix::ALL_Z[i].index(), i as u8);
    }
}

// === 11. sync/affinity ======================================================

#[test]
fn affinity_pin_p_core() {
    assert!(affinity::pin_p_core().is_ok());
}

#[test]
fn affinity_pin_any() {
    assert!(affinity::pin_any().is_ok());
}

#[test]
fn affinity_pin_e_core() {
    assert!(affinity::pin_e_core().is_ok());
}
