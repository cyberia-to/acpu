//! acpu full driver audit vs Apple frameworks.
use std::time::Instant;

fn med(t: &mut [u64]) -> u64 {
    t.sort();
    t[t.len() / 2]
}

fn ns<F: FnMut()>(mut f: F) -> u64 {
    let dl = Instant::now() + std::time::Duration::from_secs(2);
    f();
    let mut t = Vec::with_capacity(100);
    for _ in 0..100 {
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

fn row(op: &str, n: usize, acpu_ns: u64, apple_ns: u64) {
    let a = n as f64 / acpu_ns as f64;
    if apple_ns == 0 || apple_ns == u64::MAX {
        eprintln!(
            "  {:<20} {:>7} ns {:>6.1} Ge/s {:>10} {:>10}",
            op, acpu_ns, a, "—", "—"
        );
    } else {
        let ap = n as f64 / apple_ns as f64;
        let ratio = acpu_ns as f64 / apple_ns as f64;
        let mark = if ratio <= 1.05 {
            "WIN"
        } else if ratio <= 1.2 {
            "~"
        } else {
            ""
        };
        eprintln!(
            "  {:<20} {:>7} ns {:>6.1} Ge/s {:>7} ns {:>5.1}x {}",
            op, acpu_ns, a, apple_ns, ratio, mark
        );
    }
}

fn row_gf(op: &str, acpu_gf: f64, apple_gf: f64) {
    let ratio = apple_gf / acpu_gf;
    let mark = if ratio <= 1.05 {
        "WIN"
    } else if ratio <= 1.2 {
        "~"
    } else {
        ""
    };
    eprintln!(
        "  {:<20} {:>7.0} GF {:>13} {:>7.0} GF {:>5.1}x {}",
        op, acpu_gf, "", apple_gf, ratio, mark
    );
}

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
    fn vDSP_maxv(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_vsadd(a: *const f32, ia: i64, b: *const f32, c: *mut f32, ic: i64, n: u64);
    fn vDSP_vsdiv(a: *const f32, ia: i64, b: *const f32, c: *mut f32, ic: i64, n: u64);
}

fn main() {
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(30));
        eprintln!("\n!!! 30s TIMEOUT !!!");
        std::process::exit(0);
    });

    let n = 4096usize;
    let nn = n as i32;
    let nu = n as u64;
    let src: Vec<f32> = (0..n).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
    let pos: Vec<f32> = src.iter().map(|x| x.abs() + 0.01).collect();

    let c = acpu::probe::detect();
    eprintln!(
        "=== acpu driver audit — {:?} ({}P+{}E) — {} f32 ===",
        c.chip, c.p_cores, c.e_cores, n
    );
    eprintln!(
        "  {:<20} {:>7} {:>10} {:>10} {:>10}",
        "operation", "acpu", "acpu Ge/s", "apple", "ratio"
    );
    eprintln!("  {}", "─".repeat(62));

    // ---- VECTOR MATH ----
    eprintln!("\n  VECTOR MATH");
    let mut b = src.clone();
    let mut d = vec![0f32; n];
    row(
        "exp",
        n,
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::exp(&mut b);
        }),
        ns(|| unsafe {
            vvexpf(d.as_mut_ptr(), src.as_ptr(), &nn);
        }),
    );
    row(
        "log",
        n,
        ns(|| {
            b.copy_from_slice(&pos);
            acpu::vector::math::log(&mut b);
        }),
        ns(|| unsafe {
            vvlogf(d.as_mut_ptr(), pos.as_ptr(), &nn);
        }),
    );
    row(
        "tanh",
        n,
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::tanh(&mut b);
        }),
        ns(|| unsafe {
            vvtanhf(d.as_mut_ptr(), src.as_ptr(), &nn);
        }),
    );
    row(
        "sigmoid",
        n,
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::sigmoid(&mut b);
        }),
        0,
    );
    row(
        "gelu",
        n,
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::gelu(&mut b);
        }),
        0,
    );
    row(
        "silu",
        n,
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::silu(&mut b);
        }),
        0,
    );

    // ---- REDUCTIONS ----
    eprintln!("\n  REDUCTIONS");
    let mut r = 0f32;
    row(
        "sum",
        n,
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
        n,
        ns(|| {
            std::hint::black_box(acpu::vector::reduce::dot(&src, &src));
        }),
        ns(|| unsafe {
            std::hint::black_box(cblas_sdot(nn, src.as_ptr(), 1, src.as_ptr(), 1));
        }),
    );
    row(
        "norm_l2",
        n,
        ns(|| {
            std::hint::black_box(acpu::vector::reduce::norm_l2(&src));
        }),
        ns(|| unsafe {
            std::hint::black_box(cblas_snrm2(nn, src.as_ptr(), 1));
        }),
    );
    row(
        "max",
        n,
        ns(|| {
            std::hint::black_box(acpu::vector::reduce::max(&src));
        }),
        ns(|| unsafe {
            vDSP_maxv(src.as_ptr(), 1, &mut r, nu);
            std::hint::black_box(r);
        }),
    );

    // ---- COMPOUND ----
    eprintln!("\n  COMPOUND OPS");
    row(
        "softmax",
        n,
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
    let w: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1).collect();
    let mut rm = vec![0f32; n];
    acpu::vector::softmax::rmsnorm(&mut rm, &src, &w, 1e-5); // warmup with real data
    row(
        "rmsnorm",
        n,
        ns(|| {
            acpu::vector::softmax::rmsnorm(&mut rm, &src, &w, 1e-5);
        }),
        0,
    );

    // ---- CONVERSIONS ----
    eprintln!("\n  CONVERSIONS");
    let fd: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mut f16 = vec![0u16; n];
    let mut fo = vec![0f32; n];
    let mut bf = vec![0u16; n];
    let mut i8 = vec![0i8; n];
    row(
        "f32→f16",
        n,
        ns(|| {
            acpu::cvt_f32_f16(&mut f16, &fd);
        }),
        0,
    );
    acpu::cvt_f32_f16(&mut f16, &fd);
    row(
        "f16→f32",
        n,
        ns(|| {
            acpu::cvt_f16_f32(&mut fo, &f16);
        }),
        0,
    );
    row(
        "f32→bf16",
        n,
        ns(|| {
            acpu::cvt_f32_bf16(&mut bf, &fd);
        }),
        0,
    );
    acpu::cvt_f32_bf16(&mut bf, &fd);
    row(
        "bf16→f32",
        n,
        ns(|| {
            acpu::cvt_bf16_f32(&mut fo, &bf);
        }),
        0,
    );
    row(
        "f32→i8",
        n,
        ns(|| {
            acpu::cvt_f32_i8(&mut i8, &fd, 0.1);
        }),
        0,
    );
    acpu::cvt_f32_i8(&mut i8, &fd, 0.1);
    row(
        "i8→f32",
        n,
        ns(|| {
            acpu::cvt_i8_f32(&mut fo, &i8, 0.1, 0);
        }),
        0,
    );

    // ---- GEMM ----
    eprintln!("\n  GEMM (sgemm)");
    let ag = std::thread::spawn(|| {
        let mut r = Vec::new();
        for &sz in &[64, 256, 1024] {
            let a = vec![1f32; sz * sz];
            let b = vec![1f32; sz * sz];
            let mut c = vec![0f32; sz * sz];
            let it = if sz >= 1024 { 3 } else { 10 };
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
                    c.as_mut_ptr(),
                    sz as i32,
                );
            }
            let mut t = vec![0u64; it];
            for i in 0..it {
                c.fill(0.0);
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
                        c.as_mut_ptr(),
                        sz as i32,
                    );
                }
                t[i] = s.elapsed().as_nanos() as u64;
            }
            r.push((sz, med(&mut t)));
        }
        r
    })
    .join()
    .unwrap();
    for &sz in &[64, 256, 1024] {
        let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..sz * sz).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c = vec![0f32; sz * sz];
        let it = if sz >= 1024 { 3 } else { 10 };
        acpu::sgemm(&a, &b, &mut c, sz, sz, sz);
        let mut t = vec![0u64; it];
        for i in 0..it {
            c.fill(0.0);
            let s = Instant::now();
            acpu::sgemm(&a, &b, &mut c, sz, sz, sz);
            t[i] = s.elapsed().as_nanos() as u64;
        }
        let an = med(&mut t);
        let ac = ag.iter().find(|r| r.0 == sz).unwrap().1;
        let ops = 2.0 * (sz as f64).powi(3);
        row_gf(&format!("sgemm {}", sz), ops / an as f64, ops / ac as f64);
    }

    // ---- AMX RAW ----
    eprintln!("\n  AMX RAW (per instruction)");
    eprintln!(
        "  {:<20} {:>7} {:>10}",
        "instruction", "ns/op", "peak GFLOPS"
    );
    eprintln!("  {}", "─".repeat(40));
    {
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
        macro_rules! amx {
            ($nm:expr,$op:expr,$v:expr,$fl:expr) => {
                for i in 0..it {
                    let s = Instant::now();
                    for _ in 0..ops {
                        acpu::matrix::asm::amx_op::<$op>($v);
                    }
                    t[i] = s.elapsed().as_nanos() as u64;
                }
                let n = med(&mut t);
                let npo = n as f64 / ops as f64;
                if $fl > 0 {
                    eprintln!("  {:<20} {:>7.1} {:>10.0}", $nm, npo, $fl as f64 / npo);
                } else {
                    eprintln!("  {:<20} {:>7.1}", $nm, npo);
                }
            };
        }
        use acpu::matrix::{
            asm::*,
            fma::fma_acc,
            regs::{XRow, YRow},
        };
        let f = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
        amx!("LDX", { OP_LDX }, p as u64, 0);
        amx!("LDX pair (128B)", { OP_LDX }, (p as u64) | (1u64 << 62), 0);
        amx!("LDY", { OP_LDY }, p as u64, 0);
        amx!("LDZ", { OP_LDZ }, p as u64, 0);
        amx!("STZ", { OP_STZ }, p as u64, 0);
        amx!("FMA32 16×16", { OP_FMA32 }, f, 512);
        amx!("FMA16 32×32", { OP_FMA16 }, f, 2048);
        amx!("FMA64 8×8", { OP_FMA64 }, f, 128);
        amx!("MAC16 i16", { OP_MAC16 }, f, 2048);
    }

    // ---- SYNC ----
    eprintln!("\n  SYNC PRIMITIVES");
    eprintln!("  {:<20} {:>7}", "primitive", "ns/op");
    eprintln!("  {}", "─".repeat(30));
    eprintln!(
        "  {:<20} {:>7}",
        "DMB ISH",
        ns(|| unsafe {
            acpu::sync::dmb_ish();
        })
    );
    eprintln!(
        "  {:<20} {:>7}",
        "DSB ISH",
        ns(|| unsafe {
            acpu::sync::dsb_ish();
        })
    );
    eprintln!(
        "  {:<20} {:>7}",
        "ISB",
        ns(|| unsafe {
            acpu::sync::isb();
        })
    );

    // ---- MEMORY ----
    eprintln!("\n  MEMORY BANDWIDTH (NEON sum)");
    eprintln!("  {:<20} {:>7} {:>10}", "level", "ns", "GB/s");
    eprintln!("  {}", "─".repeat(40));
    for &(sz, label) in &[(4096, "L1 16KB"), (65536, "L2 256KB"), (1048576, "L3 4MB")] {
        let data: Vec<f32> = vec![1.0; sz];
        let t = ns(|| {
            std::hint::black_box(acpu::vector::reduce::sum(&data));
        });
        let gbs = sz as f64 * 4.0 / t as f64;
        eprintln!("  {:<20} {:>7} {:>10.1}", label, t, gbs);
    }

    eprintln!("\n=== done ===");
}
