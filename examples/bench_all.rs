//! Comprehensive benchmark: acpu vs Apple frameworks across ALL operations.
//! Every category compares against the Apple equivalent.
use std::time::Instant;

fn median(t: &mut [u64]) -> u64 {
    t.sort();
    t[t.len() / 2]
}

fn bench<F: FnMut()>(label: &str, max_iters: usize, elem: usize, mut f: F) {
    let deadline = Instant::now() + std::time::Duration::from_secs(3);
    f();
    let mut times = Vec::with_capacity(max_iters.min(200));
    for _ in 0..max_iters.min(200) {
        if Instant::now() > deadline {
            break;
        }
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_nanos() as u64);
    }
    if times.is_empty() {
        eprintln!("  {:<42} TIMEOUT", label);
        return;
    }
    times.sort();
    let ns = times[times.len() / 2];
    let tp = if ns > 0 {
        elem as f64 / ns as f64
    } else {
        f64::INFINITY
    };
    eprintln!("  {:<42} {:>7} ns {:>7.2} Ge/s", label, ns, tp);
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
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn cblas_sdot(n: i32, x: *const f32, incx: i32, y: *const f32, incy: i32) -> f32;
    fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32;
    fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvlogf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
    fn vDSP_sve(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_maxv(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_minv(a: *const f32, ia: i64, c: *mut f32, n: u64);
    fn vDSP_dotpr(a: *const f32, ia: i64, b: *const f32, ib: i64, c: *mut f32, n: u64);
    fn vDSP_vadd(a: *const f32, ia: i64, b: *const f32, ib: i64, c: *mut f32, ic: i64, n: u64);
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
    let it = 200;

    let caps = acpu::probe::detect();
    eprintln!(
        "=== acpu driver audit — {:?} ({}P+{}E) ===\n",
        caps.chip, caps.p_cores, caps.e_cores
    );

    // ---- VECTOR MATH ----
    eprintln!("--- Vector Math ({} f32) ---            acpu      Apple", n);
    let mut buf = src.clone();
    let mut d = vec![0f32; n];

    bench("exp              acpu", it, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::math::exp(&mut buf);
    });
    bench("exp              vvexpf", it, n, || unsafe {
        vvexpf(d.as_mut_ptr(), src.as_ptr(), &nn);
    });

    let pos: Vec<f32> = src.iter().map(|x| x.abs() + 0.01).collect();
    bench("log              acpu", it, n, || {
        buf.copy_from_slice(&pos);
        acpu::vector::math::log(&mut buf);
    });
    bench("log              vvlogf", it, n, || unsafe {
        vvlogf(d.as_mut_ptr(), pos.as_ptr(), &nn);
    });

    bench("tanh             acpu", it, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::math::tanh(&mut buf);
    });
    bench("tanh             vvtanhf", it, n, || unsafe {
        vvtanhf(d.as_mut_ptr(), src.as_ptr(), &nn);
    });

    bench("sigmoid          acpu", it, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::math::sigmoid(&mut buf);
    });
    bench("gelu             acpu", it, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::math::gelu(&mut buf);
    });
    bench("silu             acpu", it, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::math::silu(&mut buf);
    });

    // ---- REDUCTIONS ----
    eprintln!(
        "\n--- Reductions ({} f32) ---              acpu      Apple",
        n
    );
    let mut r = 0f32;
    bench("sum              acpu", it, n, || {
        std::hint::black_box(acpu::vector::reduce::sum(&src));
    });
    bench("sum              vDSP_sve", it, n, || unsafe {
        vDSP_sve(src.as_ptr(), 1, &mut r, nu);
        std::hint::black_box(r);
    });

    bench("dot              acpu", it, n, || {
        std::hint::black_box(acpu::vector::reduce::dot(&src, &src));
    });
    bench("dot              cblas_sdot", it, n, || unsafe {
        std::hint::black_box(cblas_sdot(nn, src.as_ptr(), 1, src.as_ptr(), 1));
    });

    bench("norm_l2          acpu", it, n, || {
        std::hint::black_box(acpu::vector::reduce::norm_l2(&src));
    });
    bench("norm_l2          cblas_snrm2", it, n, || unsafe {
        std::hint::black_box(cblas_snrm2(nn, src.as_ptr(), 1));
    });

    bench("max              acpu", it, n, || {
        std::hint::black_box(acpu::vector::reduce::max(&src));
    });
    bench("max              vDSP_maxv", it, n, || unsafe {
        vDSP_maxv(src.as_ptr(), 1, &mut r, nu);
        std::hint::black_box(r);
    });

    // ---- COMPOUND ----
    eprintln!(
        "\n--- Compound Ops ({} f32) ---            acpu      Apple",
        n
    );
    bench("softmax          acpu", it, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::softmax::softmax(&mut buf);
    });
    // vDSP softmax pipeline: maxv + vsadd(-max) + vvexpf + sve + vsdiv
    bench("softmax          vDSP pipeline", it, n, || unsafe {
        let mut mx = 0f32;
        vDSP_maxv(src.as_ptr(), 1, &mut mx, nu);
        let neg_mx = -mx;
        vDSP_vsadd(src.as_ptr(), 1, &neg_mx, d.as_mut_ptr(), 1, nu);
        vvexpf(d.as_mut_ptr(), d.as_ptr(), &nn);
        let mut s = 0f32;
        vDSP_sve(d.as_ptr(), 1, &mut s, nu);
        vDSP_vsdiv(d.as_ptr(), 1, &s, d.as_mut_ptr(), 1, nu);
    });

    let w: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1).collect();
    let mut rms = vec![0f32; n];
    bench("rmsnorm          acpu", it, n, || {
        acpu::vector::softmax::rmsnorm(&mut rms, &src, &w, 1e-5);
    });

    // ---- CONVERSIONS ----
    eprintln!("\n--- Conversions ({} elem) ---", n);
    let f32d: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mut f16b = vec![0u16; n];
    let mut f32o = vec![0f32; n];
    let mut bf16b = vec![0u16; n];
    let mut i8b = vec![0i8; n];

    bench("f32→f16          acpu", it, n, || {
        acpu::cvt_f32_f16(&mut f16b, &f32d);
    });
    acpu::cvt_f32_f16(&mut f16b, &f32d);
    bench("f16→f32          acpu", it, n, || {
        acpu::cvt_f16_f32(&mut f32o, &f16b);
    });
    bench("f32→bf16         acpu", it, n, || {
        acpu::cvt_f32_bf16(&mut bf16b, &f32d);
    });
    acpu::cvt_f32_bf16(&mut bf16b, &f32d);
    bench("bf16→f32         acpu", it, n, || {
        acpu::cvt_bf16_f32(&mut f32o, &bf16b);
    });
    bench("f32→i8           acpu", it, n, || {
        acpu::cvt_f32_i8(&mut i8b, &f32d, 0.1);
    });
    acpu::cvt_f32_i8(&mut i8b, &f32d, 0.1);
    bench("i8→f32           acpu", it, n, || {
        acpu::cvt_i8_f32(&mut f32o, &i8b, 0.1, 0);
    });

    // ---- GEMM ----
    eprintln!("\n--- GEMM ---                             acpu      Accelerate");
    let accel = std::thread::spawn(|| {
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
            r.push((sz, median(&mut t)));
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
        let ans = median(&mut t);
        let acc = accel.iter().find(|r| r.0 == sz).unwrap().1;
        let ops = 2.0 * (sz as f64).powi(3);
        let ratio = ans as f64 / acc as f64;
        let status = if ratio < 1.05 { "★" } else { " " };
        eprintln!(
            "  sgemm {:>5}  {:>7.0} GF {:>7.0} GF  {:.2}x {}",
            sz,
            ops / ans as f64,
            ops / acc as f64,
            ratio,
            status
        );
    }

    // ---- AMX RAW ----
    eprintln!("\n--- AMX Raw Throughput ---                ns/op  peak GFLOPS");
    {
        let ta = [1f32; 256];
        let tb = [1f32; 256];
        let mut tc = [0f32; 256];
        acpu::sgemm(&ta, &tb, &mut tc, 16, 16, 16);
    }
    unsafe {
        #[repr(align(128))]
        struct A128([f32; 64]);
        let mut b = A128([1.0; 64]);
        let p = b.0.as_mut_ptr() as *mut u8;
        let it = 100;
        let ops = 50;
        let mut t = vec![0u64; it];
        macro_rules! amx {
            ($name:expr, $op:expr, $operand:expr, $flops:expr) => {
                for i in 0..it {
                    let s = Instant::now();
                    for _ in 0..ops {
                        acpu::matrix::asm::amx_op::<$op>($operand);
                    }
                    t[i] = s.elapsed().as_nanos() as u64;
                }
                let ns = median(&mut t);
                let npo = ns as f64 / ops as f64;
                if $flops > 0 {
                    eprintln!(
                        "  {:<38} {:>5.1}   {:>7.0}",
                        $name,
                        npo,
                        $flops as f64 / npo
                    );
                } else {
                    eprintln!("  {:<38} {:>5.1}", $name, npo);
                }
            };
        }
        use acpu::matrix::{
            asm::*,
            fma::fma_acc,
            regs::{XRow, YRow},
        };
        let f = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
        amx!("LDX (64B)", { OP_LDX }, p as u64, 0);
        amx!("LDX pair (128B)", { OP_LDX }, (p as u64) | (1u64 << 62), 0);
        amx!("LDY (64B)", { OP_LDY }, p as u64, 0);
        amx!("LDZ (64B)", { OP_LDZ }, p as u64, 0);
        amx!("STZ (64B)", { OP_STZ }, p as u64, 0);
        amx!("FMA32 (16×16=512 FLOPS)", { OP_FMA32 }, f, 512);
        amx!("FMA16 (32×32=2048 FLOPS)", { OP_FMA16 }, f, 2048);
        amx!("FMA64 (8×8=128 FLOPS)", { OP_FMA64 }, f, 128);
        amx!("MAC16 (i16 32×32)", { OP_MAC16 }, f, 2048);
        amx!("FMS32 (16×16 subtract)", { OP_FMS32 }, f, 512);
    }

    // ---- SYNC ----
    eprintln!("\n--- Sync Primitives ---                   ns/op");
    bench("DMB ISH", 10000, 1, || unsafe {
        acpu::sync::dmb_ish();
    });
    bench("DSB ISH", 10000, 1, || unsafe {
        acpu::sync::dsb_ish();
    });
    bench("ISB", 10000, 1, || unsafe {
        acpu::sync::isb();
    });

    // ---- MEMORY BW ----
    eprintln!("\n--- Memory Bandwidth ---                  GB/s");
    for &(sz, label) in &[(4096, "L1 16KB"), (65536, "L2 256KB"), (1048576, "L3 4MB")] {
        let data: Vec<f32> = vec![1.0; sz];
        bench(label, 100, sz, || {
            std::hint::black_box(acpu::vector::reduce::sum(&data));
        });
    }

    eprintln!("\n=== done ===");
}
