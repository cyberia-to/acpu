// ---------------------------------------------------------------------------
// Elementwise math functions -- NEON fast-path + scalar fallback
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ---- constants ------------------------------------------------------------

const LN2: f32 = core::f32::consts::LN_2; // 0.6931472
const LN2_INV: f32 = 1.0 / LN2; // 1.442695
const LN2_HI: f32 = 0.693_145_75; // high part for Cody-Waite
const LN2_LO: f32 = 1.428_606_8e-6; // low part

// Polynomial coefficients for exp(r) on [-ln2/2, ln2/2]  (Cephes / Remez)
const EXP_P0: f32 = 1.0;
const EXP_P1: f32 = 1.0;
const EXP_P2: f32 = 0.5; // 1/2!
const EXP_P3: f32 = 0.166_666_7; // 1/3!
const EXP_P4: f32 = 0.041_666_67; // 1/4!
const EXP_P5: f32 = 0.008_333_34; // 1/5!
const EXP_P6: f32 = 0.001_388_89; // 1/6!

const EXP_HI: f32 = 88.376_26;
const EXP_LO: f32 = -87.336_55;

// GELU constant  sqrt(2/pi)
const SQRT_2_OVER_PI: f32 = 0.797_884_6;
const GELU_COEFF: f32 = 0.044_715;

// ---- scalar helpers -------------------------------------------------------

#[inline(always)]
fn exp_scalar(x: f32) -> f32 {
    let x = x.clamp(EXP_LO, EXP_HI);
    let n = (x * LN2_INV + 0.5).floor();
    let r = x - n * LN2_HI - n * LN2_LO;
    // Horner form: p = P0 + r*(P1 + r*(P2 + r*(P3 + r*(P4 + r*(P5 + r*P6)))))
    let mut p = EXP_P6;
    p = p * r + EXP_P5;
    p = p * r + EXP_P4;
    p = p * r + EXP_P3;
    p = p * r + EXP_P2;
    p = p * r + EXP_P1;
    p = p * r + EXP_P0;
    // multiply by 2^n via bit manipulation
    let ni = n as i32;
    let bits = ((ni + 127) as u32) << 23;
    let pow2n = f32::from_bits(bits);
    p * pow2n
}

#[inline(always)]
fn log_scalar(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    let m_bits = (bits & 0x007F_FFFF) | 0x3F00_0000; // mantissa in [0.5, 1)
    let mut m = f32::from_bits(m_bits);
    let mut exp_adj = e as f32;
    if m < core::f32::consts::FRAC_1_SQRT_2 {
        m *= 2.0;
        exp_adj -= 1.0;
    }
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let s2 = s * s;
    let s4 = s2 * s2;
    let t1 = s2 * (0.666_666_6 + s4 * (0.285_714_3 + s4 * 0.206_349_21));
    let t2 = s4 * (0.4 + s4 * (0.222_222_22 + s4 * 0.153_846_15));
    let r = t1 + t2;
    let hfsq = 0.5 * f * f;
    exp_adj * LN2 + (s * (hfsq + r) + (f - hfsq))
}

#[inline(always)]
fn tanh_scalar(x: f32) -> f32 {
    if x.abs() > 9.0 {
        return x.signum();
    }
    let e2x = exp_scalar(2.0 * x);
    (e2x - 1.0) / (e2x + 1.0)
}

#[inline(always)]
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + exp_scalar(-x))
}

// ---- NEON helpers ---------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn exp_neon(x: float32x4_t) -> float32x4_t {
    let lo = vdupq_n_f32(EXP_LO);
    let hi = vdupq_n_f32(EXP_HI);
    let x = vmaxq_f32(vminq_f32(x, hi), lo);

    let inv_ln2 = vdupq_n_f32(LN2_INV);
    let half = vdupq_n_f32(0.5);
    // n = floor(x / ln2 + 0.5)
    let n = vrndmq_f32(vaddq_f32(vmulq_f32(x, inv_ln2), half));

    // Cody-Waite reduction: r = x - n*LN2_HI - n*LN2_LO
    let ln2_hi = vdupq_n_f32(LN2_HI);
    let ln2_lo = vdupq_n_f32(LN2_LO);
    let r = vsubq_f32(vsubq_f32(x, vmulq_f32(n, ln2_hi)), vmulq_f32(n, ln2_lo));

    // Horner polynomial
    let mut p = vdupq_n_f32(EXP_P6);
    p = vfmaq_f32(vdupq_n_f32(EXP_P5), p, r);
    p = vfmaq_f32(vdupq_n_f32(EXP_P4), p, r);
    p = vfmaq_f32(vdupq_n_f32(EXP_P3), p, r);
    p = vfmaq_f32(vdupq_n_f32(EXP_P2), p, r);
    p = vfmaq_f32(vdupq_n_f32(EXP_P1), p, r);
    p = vfmaq_f32(vdupq_n_f32(EXP_P0), p, r);

    // 2^n via integer add to exponent field
    let ni = vcvtq_s32_f32(n);
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(ni, vdupq_n_s32(127))));
    vmulq_f32(p, pow2n)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn log_neon(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let inv_sqrt2 = vdupq_n_f32(core::f32::consts::FRAC_1_SQRT_2);
    let ln2_v = vdupq_n_f32(LN2);

    let bits = vreinterpretq_s32_f32(x);
    // exponent
    let e_raw = vsubq_s32(
        vshrq_n_s32::<23>(vandq_s32(bits, vdupq_n_s32(0x7F80_0000u32 as i32))),
        vdupq_n_s32(127),
    );
    // mantissa in [0.5, 1)
    let m_bits = vorrq_s32(
        vandq_s32(bits, vdupq_n_s32(0x007F_FFFFu32 as i32)),
        vdupq_n_s32(0x3F00_0000u32 as i32),
    );
    let m = vreinterpretq_f32_s32(m_bits);

    // adjust if m < 1/sqrt(2)
    let mask = vcltq_f32(m, inv_sqrt2);
    let m = vbslq_f32(mask, vmulq_f32(m, vdupq_n_f32(2.0)), m);
    let e_adj = vbslq_s32(mask, vsubq_s32(e_raw, vdupq_n_s32(1)), e_raw);

    let f = vsubq_f32(m, one);
    let s = vdivq_f32(f, vaddq_f32(vdupq_n_f32(2.0), f));
    let s2 = vmulq_f32(s, s);
    let s4 = vmulq_f32(s2, s2);

    let mut t1 = vfmaq_f32(vdupq_n_f32(0.285_714_3), vdupq_n_f32(0.206_349_21), s4);
    t1 = vfmaq_f32(vdupq_n_f32(0.666_666_6), t1, s4);
    t1 = vmulq_f32(t1, s2);

    let mut t2 = vfmaq_f32(vdupq_n_f32(0.222_222_22), vdupq_n_f32(0.153_846_15), s4);
    t2 = vfmaq_f32(vdupq_n_f32(0.4), t2, s4);
    t2 = vmulq_f32(t2, s4);

    let r = vaddq_f32(t1, t2);
    let hfsq = vmulq_f32(vdupq_n_f32(0.5), vmulq_f32(f, f));
    let ef = vcvtq_f32_s32(e_adj);
    vaddq_f32(
        vmulq_f32(ef, ln2_v),
        vaddq_f32(vmulq_f32(s, vaddq_f32(hfsq, r)), vsubq_f32(f, hfsq)),
    )
}

// ---- public functions -----------------------------------------------------

/// Elementwise e^x in-place.
pub fn exp(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                let r = exp_neon(v);
                vst1q_f32(x.as_mut_ptr().add(i), r);
                i += 4;
            }
        }
    }

    // scalar tail / fallback
    while i < len {
        x[i] = exp_scalar(x[i]);
        i += 1;
    }
}

/// Elementwise ln(x) in-place.
pub fn log(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                let r = log_neon(v);
                vst1q_f32(x.as_mut_ptr().add(i), r);
                i += 4;
            }
        }
    }

    while i < len {
        x[i] = log_scalar(x[i]);
        i += 1;
    }
}

/// Elementwise tanh in-place.
pub fn tanh(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let two = vdupq_n_f32(2.0);
            let one = vdupq_n_f32(1.0);
            let clamp_pos = vdupq_n_f32(9.0);
            let clamp_neg = vdupq_n_f32(-9.0);
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                // clamp to avoid overflow
                let clamped = vmaxq_f32(vminq_f32(v, clamp_pos), clamp_neg);
                let e2x = exp_neon(vmulq_f32(two, clamped));
                let r = vdivq_f32(vsubq_f32(e2x, one), vaddq_f32(e2x, one));
                vst1q_f32(x.as_mut_ptr().add(i), r);
                i += 4;
            }
        }
    }

    while i < len {
        x[i] = tanh_scalar(x[i]);
        i += 1;
    }
}

/// Elementwise sigmoid 1/(1+e^-x) in-place.
pub fn sigmoid(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let one = vdupq_n_f32(1.0);
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                let neg = vnegq_f32(v);
                let e = exp_neon(neg);
                let denom = vaddq_f32(one, e);
                let r = vdivq_f32(one, denom);
                vst1q_f32(x.as_mut_ptr().add(i), r);
                i += 4;
            }
        }
    }

    while i < len {
        x[i] = sigmoid_scalar(x[i]);
        i += 1;
    }
}

/// Elementwise GELU in-place.
/// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let half = vdupq_n_f32(0.5);
            let one = vdupq_n_f32(1.0);
            let two = vdupq_n_f32(2.0);
            let coeff = vdupq_n_f32(GELU_COEFF);
            let s2pi = vdupq_n_f32(SQRT_2_OVER_PI);
            let clamp_pos = vdupq_n_f32(9.0);
            let clamp_neg = vdupq_n_f32(-9.0);

            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                let x3 = vmulq_f32(vmulq_f32(v, v), v);
                let inner = vmulq_f32(s2pi, vfmaq_f32(v, coeff, x3));
                // tanh(inner) via (e^(2*inner)-1)/(e^(2*inner)+1)
                let inner_c = vmaxq_f32(vminq_f32(inner, clamp_pos), clamp_neg);
                let e2 = exp_neon(vmulq_f32(two, inner_c));
                let th = vdivq_f32(vsubq_f32(e2, one), vaddq_f32(e2, one));
                let r = vmulq_f32(half, vmulq_f32(v, vaddq_f32(one, th)));
                vst1q_f32(x.as_mut_ptr().add(i), r);
                i += 4;
            }
        }
    }

    while i < len {
        let xi = x[i];
        let inner = SQRT_2_OVER_PI * (xi + GELU_COEFF * xi * xi * xi);
        x[i] = 0.5 * xi * (1.0 + tanh_scalar(inner));
        i += 1;
    }
}

/// Elementwise SiLU (x * sigmoid(x)) in-place.
pub fn silu(x: &mut [f32]) {
    let len = x.len();
    let mut i = 0;

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let one = vdupq_n_f32(1.0);
            while i + 4 <= len {
                let v = vld1q_f32(x.as_ptr().add(i));
                let neg = vnegq_f32(v);
                let e = exp_neon(neg);
                let sig = vdivq_f32(one, vaddq_f32(one, e));
                let r = vmulq_f32(v, sig);
                vst1q_f32(x.as_mut_ptr().add(i), r);
                i += 4;
            }
        }
    }

    while i < len {
        let xi = x[i];
        x[i] = xi * sigmoid_scalar(xi);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_exp_basic() {
        let mut v = vec![0.0, 1.0, -1.0, 2.0];
        exp(&mut v);
        assert!(approx_eq(v[0], 1.0, 1e-5));
        assert!(approx_eq(v[1], std::f32::consts::E, 1e-4));
        assert!(approx_eq(v[2], 1.0 / std::f32::consts::E, 1e-4));
    }

    #[test]
    fn test_sigmoid_bounds() {
        let mut v = vec![-10.0, 0.0, 10.0];
        sigmoid(&mut v);
        assert!(v[0] < 0.01);
        assert!(approx_eq(v[1], 0.5, 1e-5));
        assert!(v[2] > 0.99);
    }

    #[test]
    fn test_tanh_bounds() {
        let mut v = vec![-20.0, 0.0, 20.0];
        tanh(&mut v);
        assert!(approx_eq(v[0], -1.0, 1e-5));
        assert!(approx_eq(v[1], 0.0, 1e-5));
        assert!(approx_eq(v[2], 1.0, 1e-5));
    }

    #[test]
    fn test_log_basic() {
        let mut v = vec![1.0, std::f32::consts::E, 0.5];
        log(&mut v);
        assert!(approx_eq(v[0], 0.0, 1e-5));
        assert!(approx_eq(v[1], 1.0, 1e-4));
        assert!(approx_eq(v[2], -(2.0f32.ln()), 1e-4));
    }

    #[test]
    fn test_gelu_zero() {
        let mut v = vec![0.0];
        gelu(&mut v);
        assert!(approx_eq(v[0], 0.0, 1e-6));
    }

    #[test]
    fn test_silu_zero() {
        let mut v = vec![0.0];
        silu(&mut v);
        assert!(approx_eq(v[0], 0.0, 1e-6));
    }
}
