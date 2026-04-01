//! acpu vs Apple Accelerate — comprehensive benchmark.
//! every row: operation | acpu | accelerate | speedup (>1× = acpu wins)
use std::time::Instant;

fn med(t: &mut [u64]) -> u64 {
    t.sort();
    t[t.len() / 2]
}

fn ns<F: FnMut()>(mut f: F) -> u64 {
    f();
    let dl = Instant::now() + std::time::Duration::from_secs(2);
    let mut t = Vec::with_capacity(200);
    for _ in 0..200 {
        if Instant::now() > dl {
            break;
        }
        let s = Instant::now();
        f();
        t.push(s.elapsed().as_nanos() as u64);
    }
    if t.is_empty() {
        return u64::MAX;
    }
    med(&mut t)
}

// ── output formatting ────────────────────────────────────────────────────

static mut WINS: u32 = 0;
static mut TIES: u32 = 0;
static mut TOTAL: u32 = 0;

fn hdr(section: &str) {
    eprintln!("\n  {section}");
    eprintln!(
        "  {:<18} {:>9} {:>9} {:>8}",
        "operation", "acpu", "apple", "speedup"
    );
    eprintln!("  {}", "─".repeat(48));
}

fn row(op: &str, acpu_ns: u64, apple_ns: u64) {
    unsafe {
        TOTAL += 1;
    }
    if apple_ns == 0 || apple_ns == u64::MAX {
        eprintln!("  {:<18} {:>8}ns {:>9} {:>8}", op, acpu_ns, "—", "—");
    } else {
        let speedup = apple_ns as f64 / acpu_ns as f64;
        let marker = if speedup >= 1.05 {
            unsafe { WINS += 1 };
            " ←"
        } else if speedup <= 0.95 {
            ""
        } else {
            unsafe { TIES += 1 };
            "  ≈"
        };
        eprintln!(
            "  {:<18} {:>8}ns {:>8}ns {:>6.2}×{}",
            op, acpu_ns, apple_ns, speedup, marker
        );
    }
}

fn row_gf(op: &str, acpu_gf: f64, apple_gf: f64) {
    unsafe {
        TOTAL += 1;
    }
    let ratio = acpu_gf / apple_gf;
    let marker = if ratio >= 1.05 {
        unsafe { WINS += 1 };
        " ←"
    } else if ratio <= 0.95 {
        ""
    } else {
        unsafe { TIES += 1 };
        "  ≈"
    };
    eprintln!(
        "  {:<18} {:>7.0}GF {:>7.0}GF {:>6.2}×{}",
        op, acpu_gf, apple_gf, ratio, marker
    );
}

// ── Apple Accelerate FFI ─────────────────────────────────────────────────

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        o: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        a: f32,
        ap: *const f32,
        lda: i32,
        bp: *const f32,
        ldb: i32,
        b: f32,
        cp: *mut f32,
        ldc: i32,
    );
    fn cblas_sdot(n: i32, x: *const f32, ix: i32, y: *const f32, iy: i32) -> f32;
    fn cblas_snrm2(n: i32, x: *const f32, ix: i32) -> f32;
    fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvlogf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);

    fn vDSP_sve(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_svesq(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_maxv(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_minv(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_vneg(a: *const f32, ia: i64, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vadd(a: *const f32, ia: i64, b: *const f32, ib: i64, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vmul(a: *const f32, ia: i64, b: *const f32, ib: i64, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vsadd(a: *const f32, ia: i64, b: *const f32, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vsmul(a: *const f32, ia: i64, b: *const f32, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vsdiv(a: *const f32, ia: i64, b: *const f32, c: *mut f32, ic: i64, n: u64);
    fn vDSP_svdiv(a: *const f32, b: *const f32, ib: i64, c: *mut f32, ic: i64, n: u64);
}

fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(120));
        eprintln!("\n!!! 120s TIMEOUT !!!");
        std::process::exit(0);
    });

    let n = 4096usize;
    let nn = n as i32;
    let nu = n as u64;
    let src: Vec<f32> = (0..n).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
    let pos: Vec<f32> = src.iter().map(|x| x.abs() + 0.01).collect();
    let mut b = src.clone();
    let mut d = vec![0f32; n];
    let mut d2 = vec![0f32; n];

    let c = acpu::probe::detect();
    eprintln!(
        "=== acpu vs Apple Accelerate — {:?} ({}P+{}E) — {} f32 ===",
        c.chip, c.p_cores, c.e_cores, n
    );

    // ── ELEMENTWISE ──────────────────────────────────────────────────────

    hdr("ELEMENTWISE (in-place, 4096 f32)");

    row(
        "exp",
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::exp(&mut b);
        }),
        ns(|| unsafe { vvexpf(d.as_mut_ptr(), src.as_ptr(), &nn) }),
    );
    row(
        "log",
        ns(|| {
            b.copy_from_slice(&pos);
            acpu::vector::math::log(&mut b);
        }),
        ns(|| unsafe { vvlogf(d.as_mut_ptr(), pos.as_ptr(), &nn) }),
    );
    row(
        "tanh",
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::tanh(&mut b);
        }),
        ns(|| unsafe { vvtanhf(d.as_mut_ptr(), src.as_ptr(), &nn) }),
    );
    // Apple sigmoid: 1/(1+exp(-x))  built from vDSP_vneg + vvexpf + vDSP_vsadd + vDSP_svdiv
    row(
        "sigmoid",
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::sigmoid(&mut b);
        }),
        ns(|| unsafe {
            vDSP_vneg(src.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
            vvexpf(d.as_mut_ptr(), d.as_ptr(), &nn);
            let one: f32 = 1.0;
            vDSP_vsadd(d.as_ptr(), 1, &one, d.as_mut_ptr(), 1, nu);
            let one_r: f32 = 1.0;
            vDSP_svdiv(&one_r, d.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
        }),
    );
    // Apple gelu: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x³)))
    row(
        "gelu",
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::gelu(&mut b);
        }),
        ns(|| unsafe {
            // d = x²
            vDSP_vmul(src.as_ptr(), 1, src.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
            // d = x³
            vDSP_vmul(d.as_ptr(), 1, src.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
            // d = 0.044715 * x³
            let c1: f32 = 0.044715;
            vDSP_vsmul(d.as_ptr(), 1, &c1, d.as_mut_ptr(), 1, nu);
            // d = x + 0.044715*x³
            vDSP_vadd(src.as_ptr(), 1, d.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
            // d = sqrt(2/pi) * (...)
            let c2: f32 = 0.7978845608;
            vDSP_vsmul(d.as_ptr(), 1, &c2, d.as_mut_ptr(), 1, nu);
            // d = tanh(...)
            vvtanhf(d.as_mut_ptr(), d.as_ptr(), &nn);
            // d = 1 + tanh(...)
            let one: f32 = 1.0;
            vDSP_vsadd(d.as_ptr(), 1, &one, d.as_mut_ptr(), 1, nu);
            // d2 = 0.5 * x
            let half: f32 = 0.5;
            vDSP_vsmul(src.as_ptr(), 1, &half, d2.as_mut_ptr(), 1, nu);
            // d = 0.5*x * (1+tanh(...))
            vDSP_vmul(d2.as_ptr(), 1, d.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
        }),
    );
    // Apple silu: x * sigmoid(x)
    row(
        "silu",
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::silu(&mut b);
        }),
        ns(|| unsafe {
            vDSP_vneg(src.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
            vvexpf(d.as_mut_ptr(), d.as_ptr(), &nn);
            let one: f32 = 1.0;
            vDSP_vsadd(d.as_ptr(), 1, &one, d.as_mut_ptr(), 1, nu);
            let one_r: f32 = 1.0;
            vDSP_svdiv(&one_r, d.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
            vDSP_vmul(src.as_ptr(), 1, d.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
        }),
    );

    // ── REDUCTIONS ───────────────────────────────────────────────────────

    hdr("REDUCTIONS (4096 f32 → scalar)");
    let mut r = 0f32;

    row(
        "sum",
        ns(|| {
            std::hint::black_box(acpu::vector::reduce::sum(&src));
        }),
        ns(|| unsafe {
            vDSP_sve(src.as_ptr(), 1, &mut r, nu);
            std::hint::black_box(r);
        }),
    );
    row(
        "dot",
        ns(|| {
            std::hint::black_box(acpu::vector::reduce::dot(&src, &src));
        }),
        ns(|| unsafe {
            std::hint::black_box(cblas_sdot(nn, src.as_ptr(), 1, src.as_ptr(), 1));
        }),
    );
    row(
        "norm_l2",
        ns(|| {
            std::hint::black_box(acpu::vector::reduce::norm_l2(&src));
        }),
        ns(|| unsafe {
            std::hint::black_box(cblas_snrm2(nn, src.as_ptr(), 1));
        }),
    );
    row(
        "max",
        ns(|| {
            std::hint::black_box(acpu::vector::reduce::max(&src));
        }),
        ns(|| unsafe {
            vDSP_maxv(src.as_ptr(), 1, &mut r, nu);
            std::hint::black_box(r);
        }),
    );
    row(
        "min",
        ns(|| {
            std::hint::black_box(acpu::vector::reduce::min(&src));
        }),
        ns(|| unsafe {
            vDSP_minv(src.as_ptr(), 1, &mut r, nu);
            std::hint::black_box(r);
        }),
    );

    // ── COMPOUND OPS ─────────────────────────────────────────────────────

    hdr("COMPOUND (4096 f32)");

    // Apple softmax: max → subtract → exp → sum → divide
    row(
        "softmax",
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::softmax::softmax(&mut b);
        }),
        ns(|| unsafe {
            let mut mx = 0f32;
            vDSP_maxv(src.as_ptr(), 1, &mut mx, nu);
            let neg = -mx;
            vDSP_vsadd(src.as_ptr(), 1, &neg, d.as_mut_ptr(), 1, nu);
            vvexpf(d.as_mut_ptr(), d.as_ptr(), &nn);
            let mut s = 0f32;
            vDSP_sve(d.as_ptr(), 1, &mut s, nu);
            vDSP_vsdiv(d.as_ptr(), 1, &s, d.as_mut_ptr(), 1, nu);
        }),
    );
    // Apple rmsnorm: svesq → mean → rsqrt → scale → multiply by weight
    let w: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1).collect();
    let mut rm = vec![0f32; n];
    acpu::vector::softmax::rmsnorm(&mut rm, &src, &w, 1e-5);
    row(
        "rmsnorm",
        ns(|| {
            acpu::vector::softmax::rmsnorm(&mut rm, &src, &w, 1e-5);
            std::hint::black_box(&rm);
        }),
        ns(|| unsafe {
            let mut sumsq: f32 = 0.0;
            vDSP_svesq(src.as_ptr(), 1, &mut sumsq, nu);
            let rsqrt = 1.0 / (sumsq / n as f32 + 1e-5f32).sqrt();
            vDSP_vsmul(src.as_ptr(), 1, &rsqrt, d.as_mut_ptr(), 1, nu);
            vDSP_vmul(d.as_ptr(), 1, w.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
            std::hint::black_box(&d);
        }),
    );

    // ── CONVERSIONS ──────────────────────────────────────────────────────

    eprintln!("\n  CONVERSIONS (4096 elements, ref = memcpy)");
    eprintln!(
        "  {:<18} {:>9} {:>9} {:>8}",
        "operation", "acpu", "memcpy", "overhead"
    );
    eprintln!("  {}", "─".repeat(48));

    let fd: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mut f16b = vec![0u16; n];
    let mut fo = vec![0f32; n];
    let mut bfb = vec![0u16; n];
    let mut i8b = vec![0i8; n];
    let memcpy_ns = ns(|| {
        fo.copy_from_slice(&fd);
        std::hint::black_box(&fo);
    });
    let cvt_row = |op: &str, acpu_ns: u64| {
        let overhead = acpu_ns as f64 / memcpy_ns.max(1) as f64;
        eprintln!(
            "  {:<18} {:>8}ns {:>8}ns {:>6.2}×",
            op, acpu_ns, memcpy_ns, overhead
        );
    };
    cvt_row(
        "f32→f16",
        ns(|| {
            acpu::cvt_f32_f16(&mut f16b, &fd);
            std::hint::black_box(&f16b);
        }),
    );
    acpu::cvt_f32_f16(&mut f16b, &fd);
    cvt_row(
        "f16→f32",
        ns(|| {
            acpu::cvt_f16_f32(&mut fo, &f16b);
            std::hint::black_box(&fo);
        }),
    );
    cvt_row(
        "f32→bf16",
        ns(|| {
            acpu::cvt_f32_bf16(&mut bfb, &fd);
            std::hint::black_box(&bfb);
        }),
    );
    acpu::cvt_f32_bf16(&mut bfb, &fd);
    cvt_row(
        "bf16→f32",
        ns(|| {
            acpu::cvt_bf16_f32(&mut fo, &bfb);
            std::hint::black_box(&fo);
        }),
    );
    cvt_row(
        "f32→i8",
        ns(|| {
            acpu::cvt_f32_i8(&mut i8b, &fd, 0.1);
            std::hint::black_box(&i8b);
        }),
    );
    acpu::cvt_f32_i8(&mut i8b, &fd, 0.1);
    cvt_row(
        "i8→f32",
        ns(|| {
            acpu::cvt_i8_f32(&mut fo, &i8b, 0.1, 0);
            std::hint::black_box(&fo);
        }),
    );
    eprintln!("  (memcpy {} f32 = {}ns baseline)", n, memcpy_ns);

    // ── GEMM — full spectrum 2×2 → 4096×4096 ──────────────────────────

    let gemm_sizes: Vec<usize> = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    eprintln!("\n  SGEMM — full spectrum (GFLOPS, higher = better)");
    eprintln!(
        "  {:<18} {:>9} {:>9} {:>8}",
        "size", "acpu", "apple", "ratio"
    );
    eprintln!("  {}", "─".repeat(48));

    // Apple GEMM in parallel thread
    let gsz = gemm_sizes.clone();
    let apple_gemm = std::thread::spawn(move || {
        let mut results = Vec::new();
        for &sz in &gsz {
            let a = vec![1f32; sz * sz];
            let b = vec![1f32; sz * sz];
            let mut c_buf = vec![0f32; sz * sz];
            let it = if sz >= 2048 {
                2
            } else if sz >= 512 {
                5
            } else {
                20
            };
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    111,
                    sz as i32,
                    sz as i32,
                    sz as i32,
                    1.0,
                    a.as_ptr(),
                    sz as i32,
                    b.as_ptr(),
                    sz as i32,
                    0.0,
                    c_buf.as_mut_ptr(),
                    sz as i32,
                );
            }
            let mut t = vec![0u64; it];
            for i in 0..it {
                c_buf.fill(0.0);
                let s = Instant::now();
                unsafe {
                    cblas_sgemm(
                        101,
                        111,
                        111,
                        sz as i32,
                        sz as i32,
                        sz as i32,
                        1.0,
                        a.as_ptr(),
                        sz as i32,
                        b.as_ptr(),
                        sz as i32,
                        0.0,
                        c_buf.as_mut_ptr(),
                        sz as i32,
                    );
                }
                t[i] = s.elapsed().as_nanos() as u64;
            }
            results.push((sz, med(&mut t)));
        }
        results
    })
    .join()
    .unwrap();

    for &sz in &gemm_sizes {
        let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.1).collect();
        let bm: Vec<f32> = (0..sz * sz).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c_buf = vec![0f32; sz * sz];
        let it = if sz >= 2048 {
            2
        } else if sz >= 512 {
            5
        } else {
            20
        };
        acpu::sgemm(&a, &bm, &mut c_buf, sz, sz, sz);
        let mut t = vec![0u64; it];
        for i in 0..it {
            c_buf.fill(0.0);
            let s = Instant::now();
            acpu::sgemm(&a, &bm, &mut c_buf, sz, sz, sz);
            t[i] = s.elapsed().as_nanos() as u64;
        }
        let an = med(&mut t);
        let ac = apple_gemm.iter().find(|r| r.0 == sz).unwrap().1;
        let ops = 2.0 * (sz as f64).powi(3);
        row_gf(
            &format!("sgemm {sz}×{sz}"),
            ops / an as f64,
            ops / ac as f64,
        );
    }

    // ── AMX UTILIZATION ──────────────────────────────────────────────────

    eprintln!("\n  AMX UTILIZATION (theoretical peak @ 3.228 GHz)");
    {
        // warmup AMX context
        let ta = [1f32; 256];
        let tb = [1f32; 256];
        let mut tc = [0f32; 256];
        acpu::sgemm(&ta, &tb, &mut tc, 16, 16, 16);
    }
    unsafe {
        #[repr(align(128))]
        struct A128([f32; 64]);
        let mut buf = A128([1.0; 64]);
        let p = buf.0.as_mut_ptr() as *mut u8;
        let it = 100;
        let ops = 50;
        let mut t = vec![0u64; it];
        let ghz = 3.228;

        // ── bandwidth ──
        eprintln!("  ── bandwidth ──");
        eprintln!(
            "  {:<18} {:>7} {:>9} {:>9} {:>5}",
            "instruction", "ns/op", "GB/s", "peak", "util"
        );
        eprintln!("  {}", "─".repeat(52));

        macro_rules! amx_bw {
            ($nm:expr, $op:expr, $v:expr) => {
                for i in 0..it {
                    let s = Instant::now();
                    for _ in 0..ops {
                        acpu::matrix::asm::amx_op::<$op>($v);
                    }
                    t[i] = s.elapsed().as_nanos() as u64;
                }
                let n = med(&mut t);
                let npo = n as f64 / ops as f64;
                let peak_bw = 64.0 * ghz;
                let actual_bw = 64.0 / npo;
                eprintln!(
                    "  {:<18} {:>7.1} {:>8.1} {:>8.1} {:>4.0}%",
                    $nm,
                    npo,
                    actual_bw,
                    peak_bw,
                    actual_bw / peak_bw * 100.0
                );
            };
        }

        use acpu::matrix::{
            asm::*,
            fma::fma_acc,
            regs::{XRow, YRow},
        };
        let f = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);

        amx_bw!("LDX", { OP_LDX }, p as u64);
        amx_bw!("LDX pair", { OP_LDX }, (p as u64) | (1u64 << 62));
        amx_bw!("LDY", { OP_LDY }, p as u64);
        amx_bw!("LDZ", { OP_LDZ }, p as u64);
        amx_bw!("STZ", { OP_STZ }, p as u64);

        // ── compute ──
        eprintln!("  ── compute ──");
        eprintln!(
            "  {:<18} {:>7} {:>9} {:>9} {:>5}",
            "instruction", "ns/op", "GFLOPS", "peak", "util"
        );
        eprintln!("  {}", "─".repeat(52));

        macro_rules! amx_fp {
            ($nm:expr, $op:expr, $v:expr, $fl:expr) => {
                for i in 0..it {
                    let s = Instant::now();
                    for _ in 0..ops {
                        acpu::matrix::asm::amx_op::<$op>($v);
                    }
                    t[i] = s.elapsed().as_nanos() as u64;
                }
                let n = med(&mut t);
                let npo = n as f64 / ops as f64;
                let actual = $fl as f64 / npo;
                let peak = $fl as f64 * ghz;
                eprintln!(
                    "  {:<18} {:>7.1} {:>8.0} {:>8.0} {:>4.0}%",
                    $nm,
                    npo,
                    actual,
                    peak,
                    actual / peak * 100.0
                );
            };
        }

        amx_fp!("FMA32 16×16", { OP_FMA32 }, f, 512);
        amx_fp!("FMA16 32×32", { OP_FMA16 }, f, 2048);
        amx_fp!("FMA64 8×8", { OP_FMA64 }, f, 128);
        amx_fp!("MAC16 i16", { OP_MAC16 }, f, 2048);
    }

    // ── MEMORY HIERARCHY ─────────────────────────────────────────────────

    eprintln!("\n  MEMORY HIERARCHY (acpu sum bandwidth vs memcpy)");
    eprintln!(
        "  {:<18} {:>8} {:>8} {:>8}",
        "level", "sum GB/s", "memcpy", "ratio"
    );
    eprintln!("  {}", "─".repeat(46));
    for &(sz, label) in &[
        (4096, "L1  16KB"),
        (65536, "L2  256KB"),
        (1048576, "L3  4MB"),
    ] {
        let data: Vec<f32> = vec![1.0; sz];
        let mut dst = vec![0f32; sz];
        let sum_t = ns(|| {
            std::hint::black_box(acpu::vector::reduce::sum(&data));
        });
        let cpy_t = ns(|| {
            dst.copy_from_slice(&data);
            std::hint::black_box(&dst);
        });
        let bytes = sz as f64 * 4.0;
        let sum_bw = bytes / sum_t as f64;
        let cpy_bw = bytes / cpy_t as f64;
        eprintln!(
            "  {:<18} {:>7.1} {:>7.1} {:>6.2}×",
            label,
            sum_bw,
            cpy_bw,
            sum_bw / cpy_bw
        );
    }

    // ── SUMMARY ──────────────────────────────────────────────────────────

    let (wins, ties, total) = unsafe { (WINS, TIES, TOTAL) };
    let losses = total - wins - ties;
    eprintln!("\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    eprintln!(
        "  acpu vs Apple Accelerate: {wins} wins, {ties} ties, {losses} losses ({total} total)"
    );
    eprintln!("  ← = acpu faster, ≈ = parity (±5%)");
    eprintln!("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    eprintln!("\n=== done ===");
}
