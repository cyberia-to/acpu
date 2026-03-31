//! acpu full driver audit vs Apple frameworks.
//! Every row: operation | acpu | apple/ref | ratio
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

fn hdr() {
    eprintln!(
        "  {:<16} {:>8} {:>8}  {:>8} {:>8}  {:>5}",
        "operation", "acpu", "Ge/s", "apple", "Ge/s", "ratio"
    );
    eprintln!("  {}", "─".repeat(58));
}

fn row(op: &str, n: usize, acpu: u64, apple: u64) {
    let ag = n as f64 / acpu.max(1) as f64;
    if apple == 0 || apple == u64::MAX {
        eprintln!(
            "  {:<16} {:>7}ns {:>7.1}  {:>8} {:>8}  {:>5}",
            op, acpu, ag, "—", "—", "—"
        );
    } else {
        let apg = n as f64 / apple.max(1) as f64;
        let r = acpu as f64 / apple as f64;
        eprintln!(
            "  {:<16} {:>7}ns {:>7.1}  {:>7}ns {:>7.1}  {:>4.1}x",
            op, acpu, ag, apple, apg, r
        );
    }
}

fn row_gf(op: &str, acpu_gf: f64, apple_gf: f64) {
    let r = apple_gf / acpu_gf;
    eprintln!(
        "  {:<16} {:>16.0}GF {:>16.0}GF  {:>4.1}x",
        op, acpu_gf, apple_gf, r
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
        "=== acpu driver audit — {:?} ({}P+{}E) — {} f32 ===\n",
        c.chip, c.p_cores, c.e_cores, n
    );

    // ---- VECTOR MATH ----
    eprintln!("  VECTOR MATH");
    hdr();
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
    hdr();
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
    hdr();
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
    acpu::vector::softmax::rmsnorm(&mut rm, &src, &w, 1e-5);
    row(
        "rmsnorm",
        n,
        ns(|| {
            acpu::vector::softmax::rmsnorm(&mut rm, &src, &w, 1e-5);
        }),
        0,
    );

    // ---- CONVERSIONS (compare to memcpy as baseline) ----
    eprintln!("\n  CONVERSIONS (ref=memcpy)");
    hdr();
    let fd: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mut f16b = vec![0u16; n];
    let mut fo = vec![0f32; n];
    let mut bfb = vec![0u16; n];
    let mut i8b = vec![0i8; n];
    let memcpy_ns = ns(|| {
        fo.copy_from_slice(&fd);
        std::hint::black_box(&fo);
    });
    row(
        "f32→f16",
        n,
        ns(|| {
            acpu::cvt_f32_f16(&mut f16b, &fd);
        }),
        memcpy_ns,
    );
    acpu::cvt_f32_f16(&mut f16b, &fd);
    row(
        "f16→f32",
        n,
        ns(|| {
            acpu::cvt_f16_f32(&mut fo, &f16b);
        }),
        memcpy_ns,
    );
    row(
        "f32→bf16",
        n,
        ns(|| {
            acpu::cvt_f32_bf16(&mut bfb, &fd);
        }),
        memcpy_ns,
    );
    acpu::cvt_f32_bf16(&mut bfb, &fd);
    row(
        "bf16→f32",
        n,
        ns(|| {
            acpu::cvt_bf16_f32(&mut fo, &bfb);
        }),
        memcpy_ns,
    );
    row(
        "f32→i8",
        n,
        ns(|| {
            acpu::cvt_f32_i8(&mut i8b, &fd, 0.1);
        }),
        memcpy_ns,
    );
    acpu::cvt_f32_i8(&mut i8b, &fd, 0.1);
    row(
        "i8→f32",
        n,
        ns(|| {
            acpu::cvt_i8_f32(&mut fo, &i8b, 0.1, 0);
        }),
        memcpy_ns,
    );
    eprintln!("  (memcpy 4096 f32 = {} ns baseline)", memcpy_ns);

    // ---- GEMM ----
    eprintln!("\n  GEMM");
    eprintln!(
        "  {:<16} {:>10} {:>10}  {:>5}",
        "size", "acpu GF", "apple GF", "ratio"
    );
    eprintln!("  {}", "─".repeat(44));
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

    // ---- AMX (compare to theoretical peak at 3.2 GHz) ----
    eprintln!("\n  AMX RAW (ref=theoretical peak)");
    eprintln!(
        "  {:<16} {:>8} {:>8}  {:>8} {:>5}",
        "instruction", "ns/op", "GFLOPS", "peak GF", "util"
    );
    eprintln!("  {}", "─".repeat(50));
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
        // Theoretical: 1 op/cycle at 3.228 GHz
        let ghz = 3.228;
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
                    let actual = $fl as f64 / npo;
                    let peak = $fl as f64 * ghz;
                    eprintln!(
                        "  {:<16} {:>8.1} {:>7.0}   {:>7.0} {:>4.0}%",
                        $nm,
                        npo,
                        actual,
                        peak,
                        actual / peak * 100.0
                    );
                } else {
                    let peak_bw = 64.0 * ghz; // 64 bytes/op * GHz = GB/s
                    let actual_bw = 64.0 / npo;
                    eprintln!(
                        "  {:<16} {:>8.1} {:>7.1}GB {:>7.1}GB {:>4.0}%",
                        $nm,
                        npo,
                        actual_bw,
                        peak_bw,
                        actual_bw / peak_bw * 100.0
                    );
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
        amx!("LDX pair", { OP_LDX }, (p as u64) | (1u64 << 62), 0);
        amx!("LDY", { OP_LDY }, p as u64, 0);
        amx!("LDZ", { OP_LDZ }, p as u64, 0);
        amx!("STZ", { OP_STZ }, p as u64, 0);
        amx!("FMA32 16×16", { OP_FMA32 }, f, 512);
        amx!("FMA16 32×32", { OP_FMA16 }, f, 2048);
        amx!("FMA64 8×8", { OP_FMA64 }, f, 128);
        amx!("MAC16 i16", { OP_MAC16 }, f, 2048);
    }

    // ---- SYNC (compare to atomic CAS as baseline) ----
    eprintln!("\n  SYNC (ref=atomic CAS)");
    let cas_ns = ns(|| {
        use std::sync::atomic::{AtomicU64, Ordering};
        static A: AtomicU64 = AtomicU64::new(0);
        std::hint::black_box(A.compare_exchange(0, 1, Ordering::SeqCst, Ordering::Relaxed));
    });
    eprintln!("  {:<16} {:>8} {:>8}", "primitive", "ns/op", "ref(CAS)");
    eprintln!("  {}", "─".repeat(34));
    eprintln!("  {:<16} {:>8} {:>8}", "atomic CAS", cas_ns, "baseline");
    eprintln!(
        "  {:<16} {:>8}",
        "DMB ISH",
        ns(|| unsafe {
            acpu::sync::dmb_ish();
        })
    );
    eprintln!(
        "  {:<16} {:>8}",
        "DSB ISH",
        ns(|| unsafe {
            acpu::sync::dsb_ish();
        })
    );
    eprintln!(
        "  {:<16} {:>8}",
        "ISB",
        ns(|| unsafe {
            acpu::sync::isb();
        })
    );

    // ---- MEMORY (compare to memcpy) ----
    eprintln!("\n  MEMORY BANDWIDTH (ref=memcpy)");
    eprintln!(
        "  {:<16} {:>8} {:>8}  {:>8} {:>8}",
        "level", "sum GB/s", "memcpy", "ratio", ""
    );
    eprintln!("  {}", "─".repeat(50));
    for &(sz, label) in &[(4096, "L1 16KB"), (65536, "L2 256KB"), (1048576, "L3 4MB")] {
        let data: Vec<f32> = vec![1.0; sz];
        let mut dst = vec![0f32; sz];
        let sum_t = ns(|| {
            std::hint::black_box(acpu::vector::reduce::sum(&data));
        });
        let cpy_t = ns(|| {
            dst.copy_from_slice(&data);
            std::hint::black_box(&dst);
        });
        let sum_bw = sz as f64 * 4.0 / sum_t as f64;
        let cpy_bw = sz as f64 * 4.0 / cpy_t as f64; // read+write
        eprintln!(
            "  {:<16} {:>7.1}  {:>7.1}  {:>7.2}x",
            label,
            sum_bw,
            cpy_bw,
            sum_bw / cpy_bw
        );
    }

    eprintln!("\n=== done ===");
}
