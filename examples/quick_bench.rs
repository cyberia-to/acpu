use std::time::Instant;
fn main() {
    for &n in &[2usize, 4, 8, 16, 32, 64, 128, 256, 512] {
        let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c = vec![0.0f32; n * n];
        // warmup
        for _ in 0..200 {
            c.fill(0.0);
            acpu::sgemm(&a, &b, &mut c, n, n, n);
        }
        let mut times = Vec::with_capacity(10000);
        for _ in 0..10000 {
            c.fill(0.0);
            let t = Instant::now();
            acpu::sgemm(&a, &b, &mut c, n, n, n);
            times.push(t.elapsed().as_nanos() as u64);
        }
        times.sort();
        let ns = times[5000];
        let gf = 2.0 * (n as f64).powi(3) / ns as f64;
        println!("{:>5}  {:>8} ns  {:>8.2} GFLOPS", n, ns, gf);
    }
}
