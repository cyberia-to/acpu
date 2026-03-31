//! Bench acpu vs Accelerate — small matrix focus.
//! Accelerate on separate thread to avoid AMX context conflict.
use std::time::Instant;

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

fn median_of(times: &mut [u64]) -> u64 {
    times.sort();
    times[times.len() / 2]
}

const SIZES: &[(usize, usize)] = &[
    (2, 200),
    (3, 200),
    (4, 200),
    (6, 200),
    (8, 200),
    (10, 200),
    (12, 200),
    (16, 100),
    (20, 100),
    (24, 100),
    (32, 100),
    (48, 50),
    (64, 50),
    (96, 30),
    (128, 30),
    (192, 20),
    (256, 20),
    (384, 10),
    (512, 10),
    (768, 7),
    (1024, 7),
    (2048, 5),
    (4096, 3),
];

fn make_matrices(sz: usize) -> (Vec<f32>, Vec<f32>) {
    let mut seed: u64 = 0xDEAD_BEEF;
    let mut randf = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
    };
    (
        (0..sz * sz).map(|_| randf()).collect(),
        (0..sz * sz).map(|_| randf()).collect(),
    )
}

fn main() {
    let mut acpu_results = Vec::new();
    let mut accel_results = Vec::new();

    // Phase 1: acpu.
    for &(sz, iters) in SIZES {
        let (a, b) = make_matrices(sz);
        let mut c = vec![0.0f32; sz * sz];
        acpu::sgemm(&a, &b, &mut c, sz, sz, sz);
        let mut t = vec![0u64; iters];
        for i in 0..iters {
            c.fill(0.0);
            let s = Instant::now();
            acpu::sgemm(&a, &b, &mut c, sz, sz, sz);
            t[i] = s.elapsed().as_nanos() as u64;
        }
        acpu_results.push((sz, median_of(&mut t)));
    }

    // Phase 2: Accelerate on separate thread.
    let accel_results_inner = std::thread::spawn(move || {
        let mut results = Vec::new();
        for &(sz, iters) in SIZES {
            let (a, b) = make_matrices(sz);
            let mut c = vec![0.0f32; sz * sz];
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
            let mut t = vec![0u64; iters];
            for i in 0..iters {
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
            results.push((sz, median_of(&mut t)));
        }
        results
    })
    .join()
    .unwrap();
    accel_results = accel_results_inner;

    eprintln!(
        "{:>5} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "size", "acpu_ns", "accel_ns", "acpu_gf", "accel_gf", "ratio"
    );
    eprintln!("{}", "-".repeat(52));
    for i in 0..acpu_results.len() {
        let (sz, acpu_ns) = acpu_results[i];
        let (_, acc_ns) = accel_results[i];
        let ops = 2.0 * (sz as f64).powi(3);
        let acpu_gf = ops / acpu_ns as f64;
        let accel_gf = ops / acc_ns as f64;
        let ratio = acpu_ns as f64 / acc_ns as f64;
        eprintln!(
            "{:>5} {:>8} {:>8} {:>8.2} {:>8.2} {:>5.2}x",
            sz, acpu_ns, acc_ns, acpu_gf, accel_gf, ratio
        );
    }
}
