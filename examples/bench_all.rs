//! Comprehensive benchmark: acpu vs Apple frameworks across ALL operations.
use std::time::Instant;

fn median(t: &mut [u64]) -> u64 {
    t.sort();
    t[t.len() / 2]
}

fn bench<F: FnMut()>(label: &str, iters: usize, elem: usize, mut f: F) {
    for _ in 0..10.min(iters) {
        f();
    }
    let mut times = vec![0u64; iters];
    for i in 0..iters {
        let t = Instant::now();
        f();
        times[i] = t.elapsed().as_nanos() as u64;
    }
    let ns = median(&mut times);
    let tp = if ns > 0 {
        elem as f64 / ns as f64
    } else {
        f64::INFINITY
    };
    eprintln!("  {:<40} {:>8} ns  {:>8.2} Gelem/s", label, ns, tp);
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
    fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvlogf(y: *mut f32, x: *const f32, n: *const i32);
    fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
}

fn main() {
    let n = 4096usize;
    let src: Vec<f32> = (0..n).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
    let iters = 500;

    eprintln!("=== acpu comprehensive benchmark ({} elements) ===\n", n);

    // ---- NEON VECTOR MATH (in-place) ----
    eprintln!("--- NEON Vector Math (in-place, {} floats) ---", n);
    let mut buf = src.clone();
    bench("acpu::exp", iters, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::math::exp(&mut buf);
    });
    {
        let nn = n as i32;
        let mut d = vec![0f32; n];
        bench("vDSP vvexpf (Accelerate)", iters, n, || unsafe {
            vvexpf(d.as_mut_ptr(), src.as_ptr(), &nn);
        });
    }

    let pos: Vec<f32> = src.iter().map(|x| x.abs() + 0.01).collect();
    bench("acpu::log", iters, n, || {
        buf.copy_from_slice(&pos);
        acpu::vector::math::log(&mut buf);
    });
    {
        let nn = n as i32;
        let mut d = vec![0f32; n];
        bench("vDSP vvlogf (Accelerate)", iters, n, || unsafe {
            vvlogf(d.as_mut_ptr(), pos.as_ptr(), &nn);
        });
    }

    bench("acpu::tanh", iters, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::math::tanh(&mut buf);
    });
    {
        let nn = n as i32;
        let mut d = vec![0f32; n];
        bench("vDSP vvtanhf (Accelerate)", iters, n, || unsafe {
            vvtanhf(d.as_mut_ptr(), src.as_ptr(), &nn);
        });
    }

    bench("acpu::sigmoid", iters, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::math::sigmoid(&mut buf);
    });
    bench("acpu::gelu", iters, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::math::gelu(&mut buf);
    });
    bench("acpu::silu", iters, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::math::silu(&mut buf);
    });

    // ---- NEON REDUCTIONS ----
    eprintln!("\n--- NEON Reductions ---");
    bench("acpu::sum", iters, n, || {
        std::hint::black_box(acpu::vector::reduce::sum(&src));
    });
    bench("acpu::dot", iters, n, || {
        std::hint::black_box(acpu::vector::reduce::dot(&src, &src));
    });
    bench("acpu::norm_l2", iters, n, || {
        std::hint::black_box(acpu::vector::reduce::norm_l2(&src));
    });
    bench("acpu::max", iters, n, || {
        std::hint::black_box(acpu::vector::reduce::max(&src));
    });

    // ---- NEON COMPOUND ----
    eprintln!("\n--- NEON Compound Ops ---");
    bench("acpu::softmax", iters, n, || {
        buf.copy_from_slice(&src);
        acpu::vector::softmax::softmax(&mut buf);
    });
    let w: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1).collect();
    let mut rms_out = vec![0f32; n];
    bench("acpu::rmsnorm", iters, n, || {
        acpu::vector::softmax::rmsnorm(&mut rms_out, &src, &w, 1e-5);
    });

    // ---- NUMERIC CONVERSIONS ----
    eprintln!("\n--- Numeric Conversions ---");
    let f32_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mut f16_buf = vec![0u16; n];
    let mut f32_out = vec![0f32; n];
    let mut bf16_buf = vec![0u16; n];
    let mut i8_buf = vec![0i8; n];

    bench("cvt_f32_f16", iters, n, || {
        acpu::cvt_f32_f16(&mut f16_buf, &f32_data);
    });
    acpu::cvt_f32_f16(&mut f16_buf, &f32_data);
    bench("cvt_f16_f32", iters, n, || {
        acpu::cvt_f16_f32(&mut f32_out, &f16_buf);
    });
    bench("cvt_f32_bf16", iters, n, || {
        acpu::cvt_f32_bf16(&mut bf16_buf, &f32_data);
    });
    acpu::cvt_f32_bf16(&mut bf16_buf, &f32_data);
    bench("cvt_bf16_f32", iters, n, || {
        acpu::cvt_bf16_f32(&mut f32_out, &bf16_buf);
    });
    bench("cvt_f32_i8 (scale=0.1)", iters, n, || {
        acpu::cvt_f32_i8(&mut i8_buf, &f32_data, 0.1);
    });
    acpu::cvt_f32_i8(&mut i8_buf, &f32_data, 0.1);
    bench("cvt_i8_f32 (scale=0.1)", iters, n, || {
        acpu::cvt_i8_f32(&mut f32_out, &i8_buf, 0.1, 0);
    });

    // ---- GEMM ----
    eprintln!("\n--- GEMM vs Accelerate ---");
    let accel = std::thread::spawn(|| {
        let mut r = Vec::new();
        for &sz in &[64, 256, 1024, 4096] {
            let a = vec![1f32; sz * sz];
            let b = vec![1f32; sz * sz];
            let mut c = vec![0f32; sz * sz];
            let it = if sz >= 1024 { 5 } else { 20 };
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

    for &sz in &[64, 256, 1024, 4096] {
        let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..sz * sz).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c = vec![0f32; sz * sz];
        let it = if sz >= 1024 { 5 } else { 20 };
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
        eprintln!(
            "  sgemm {:>5}  acpu:{:>8.1} GF  accel:{:>8.1} GF  {:.2}x",
            sz,
            ops / ans as f64,
            ops / acc as f64,
            ans as f64 / acc as f64
        );
    }

    // ---- AMX RAW OPS ----
    eprintln!("\n--- AMX Raw Instruction Throughput ---");
    unsafe {
        acpu::matrix::asm::amx_set();
        #[repr(align(128))]
        struct A128([f32; 64]);
        let mut b = A128([1.0; 64]);
        let p = b.0.as_mut_ptr() as *mut u8;
        let it = 500;
        let ops = 100;
        let mut t = vec![0u64; it];

        macro_rules! amx_bench {
            ($name:expr, $op:expr, $operand:expr) => {
                for i in 0..it {
                    let s = Instant::now();
                    for _ in 0..ops {
                        acpu::matrix::asm::amx_op::<$op>($operand);
                    }
                    t[i] = s.elapsed().as_nanos() as u64;
                }
                let ns = median(&mut t);
                eprintln!("  {:<40} {:>6.1} ns/op", $name, ns as f64 / ops as f64);
            };
        }
        amx_bench!("LDX (64B → X reg)", { acpu::matrix::asm::OP_LDX }, p as u64);
        amx_bench!(
            "LDX pair (128B → X[0:1])",
            { acpu::matrix::asm::OP_LDX },
            (p as u64) | (1u64 << 62)
        );
        amx_bench!("LDY (64B → Y reg)", { acpu::matrix::asm::OP_LDY }, p as u64);
        amx_bench!("LDZ (64B → Z row)", { acpu::matrix::asm::OP_LDZ }, p as u64);
        amx_bench!("STX (X reg → 64B)", { acpu::matrix::asm::OP_STX }, p as u64);
        amx_bench!("STZ (Z row → 64B)", { acpu::matrix::asm::OP_STZ }, p as u64);

        use acpu::matrix::fma::fma_acc;
        use acpu::matrix::regs::{XRow, YRow};
        let fop = fma_acc(XRow::new_unchecked(0), YRow::new_unchecked(0), 0);
        amx_bench!(
            "FMA32 (16×16, 512 FLOPS)",
            { acpu::matrix::asm::OP_FMA32 },
            fop
        );
        amx_bench!(
            "FMA16 (32×32, 2048 FLOPS)",
            { acpu::matrix::asm::OP_FMA16 },
            fop
        );
        amx_bench!(
            "FMS32 (16×16 subtract)",
            { acpu::matrix::asm::OP_FMS32 },
            fop
        );
        amx_bench!("MAC16 (i16 32×32)", { acpu::matrix::asm::OP_MAC16 }, fop);
        amx_bench!("FMA64 (8×8 f64)", { acpu::matrix::asm::OP_FMA64 }, fop);

        // Unknown opcodes 18-22
        amx_bench!("OP18 (unknown — i32 reduced?)", 18, fop);
        amx_bench!("OP19 (unknown — dot product?)", 19, fop);
        amx_bench!("OP20 (unknown — i32 matrix?)", 20, fop);
        amx_bench!("OP21 (unknown — f32 matrix?)", 21, fop);
        amx_bench!("OP22 (unknown — X extract?)", 22, fop);

        acpu::matrix::asm::amx_clr();
    }

    eprintln!("\n=== done ===");
}
