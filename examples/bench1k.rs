//! Focused 1024×1024 benchmark — compare single-thread, multi-thread, Accelerate.
//!
//! Run: cargo run --example bench1k --release

use std::time::{Duration, Instant};

const N: usize = 1024;
const ITERS: usize = 50;

fn bench_median<F: FnMut()>(mut f: F) -> Duration {
    for _ in 0..5 {
        f();
    }
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        times.push(t.elapsed());
    }
    times.sort();
    times[ITERS / 2]
}

fn gflops(dur: Duration) -> f64 {
    2.0 * (N as f64).powi(3) / dur.as_nanos() as f64
}

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
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
}

fn main() {
    let mut seed: u64 = 0xDEAD_BEEF;
    let mut randf = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };
    let a: Vec<f32> = (0..N * N).map(|_| randf()).collect();
    let b: Vec<f32> = (0..N * N).map(|_| randf()).collect();
    let mut c = vec![0.0f32; N * N];

    let caps = acpu::detect();
    println!("chip: {}, p_cores: {}", caps.chip, caps.p_cores);
    println!("matrix: {}×{}\n", N, N);

    // acpu sgemm (auto-dispatch: may use parallel)
    let dur_acpu = bench_median(|| {
        c.fill(0.0);
        acpu::sgemm(&a, &b, &mut c, N, N, N);
        std::hint::black_box(&c);
    });

    // Accelerate
    let dur_accel = bench_median(|| {
        c.fill(0.0);
        unsafe {
            cblas_sgemm(
                101,
                111,
                111,
                N as i32,
                N as i32,
                N as i32,
                1.0,
                a.as_ptr(),
                N as i32,
                b.as_ptr(),
                N as i32,
                0.0,
                c.as_mut_ptr(),
                N as i32,
            );
        }
        std::hint::black_box(&c);
    });

    let acpu_gf = gflops(dur_acpu);
    let accel_gf = gflops(dur_accel);
    let ratio = dur_acpu.as_nanos() as f64 / dur_accel.as_nanos() as f64;

    println!(
        "{:>20}  {:>8.1} µs  {:>8.1} GFLOPS",
        "acpu",
        dur_acpu.as_nanos() as f64 / 1000.0,
        acpu_gf
    );
    println!(
        "{:>20}  {:>8.1} µs  {:>8.1} GFLOPS",
        "Accelerate",
        dur_accel.as_nanos() as f64 / 1000.0,
        accel_gf
    );
    println!("{:>20}  {:>8.1}x", "ratio", ratio);
    println!(
        "{:>20}  {:>7.0}%",
        "% of Accelerate",
        acpu_gf / accel_gf * 100.0
    );
}
