//! Benchmark acpu::matmul_f32 vs Apple Accelerate cblas_sgemm.
//!
//! Run with: `cargo run --example bench_sgemm --release`

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

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

use std::time::{Duration, Instant};

const SIZES: &[usize] = &[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const ITERS: usize = 200;

/// Run `f` for `ITERS` iterations and return the median duration.
fn bench_median<F: FnMut()>(mut f: F) -> Duration {
    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed());
    }
    times.sort();
    times[ITERS / 2]
}

fn gflops(m: usize, n: usize, k: usize, dur: Duration) -> f64 {
    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    ops / dur.as_nanos() as f64 // ops / ns = GFLOP/s
}

fn main() {
    println!(
        "{:>5} | {:>10} | {:>10} | {:>8} | {:>8} | {:>5}",
        "size", "acpu_us", "accel_us", "acpu_gf", "accel_gf", "ratio"
    );
    println!("{}", "-".repeat(62));

    for &sz in SIZES {
        let m = sz;
        let n = sz;
        let k = sz;

        // Prepare matrices -- random-ish data via simple LCG to avoid
        // denormals / special patterns.
        let mut seed: u64 = 0xDEAD_BEEF;
        let mut randf = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };

        let a: Vec<f32> = (0..m * k).map(|_| randf()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| randf()).collect();

        // -- acpu --
        let acpu_dur = bench_median(|| {
            let mut c = vec![0.0f32; m * n];
            acpu::matmul_f32(&a, &b, &mut c, m, n, k);
            std::hint::black_box(&c);
        });

        // -- Accelerate --
        let accel_dur = bench_median(|| {
            let mut c = vec![0.0f32; m * n];
            unsafe {
                cblas_sgemm(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a.as_ptr(),
                    k as i32,
                    b.as_ptr(),
                    n as i32,
                    0.0,
                    c.as_mut_ptr(),
                    n as i32,
                );
            }
            std::hint::black_box(&c);
        });

        let acpu_us = acpu_dur.as_nanos() as f64 / 1000.0;
        let accel_us = accel_dur.as_nanos() as f64 / 1000.0;
        let ratio = acpu_us / accel_us;
        let acpu_gf = gflops(m, n, k, acpu_dur);
        let accel_gf = gflops(m, n, k, accel_dur);

        println!(
            "{:>5} | {:>10.1} | {:>10.1} | {:>8.2} | {:>8.2} | {:>5.2}x",
            sz, acpu_us, accel_us, acpu_gf, accel_gf, ratio
        );
    }
}
