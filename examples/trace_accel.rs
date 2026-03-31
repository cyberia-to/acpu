//! Call cblas_sgemm in a loop for profiling/sampling.
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
    let n = 64i32;
    let a = vec![1.0f32; (n * n) as usize];
    let b = vec![1.0f32; (n * n) as usize];
    let mut c = vec![0.0f32; (n * n) as usize];

    eprintln!("looping cblas_sgemm 64×64 for profiling...");
    for _ in 0..1_000_000 {
        c.fill(0.0);
        unsafe {
            cblas_sgemm(
                101,
                111,
                111,
                n,
                n,
                n,
                1.0,
                a.as_ptr(),
                n,
                b.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
            );
        }
        std::hint::black_box(&c);
    }
    eprintln!("done");
}
