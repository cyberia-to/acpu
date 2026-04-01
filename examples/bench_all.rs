//! acpu — comprehensive CPU driver benchmark.
//! hardware characterization + acpu vs Apple Accelerate comparison.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
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

// ── scoreboard ───────────────────────────────────────────────────────────

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
    unsafe { TOTAL += 1 }
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
    unsafe { TOTAL += 1 }
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

// ── pointer-chase memory latency ─────────────────────────────────────────

fn chase_latency_ns(size_bytes: usize) -> f64 {
    let n = size_bytes / std::mem::size_of::<usize>();
    let mut arr: Vec<usize> = (0..n).collect();
    // Sattolo: guaranteed single-cycle random permutation
    let mut rng: u64 = 0xdeadbeef12345678;
    for i in (1..n).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (rng >> 33) as usize % i;
        arr.swap(i, j);
    }
    let mut idx = 0usize;
    for _ in 0..n {
        idx = arr[idx];
    }
    let iters = 1_000_000usize.min(n * 10).max(n);
    let s = Instant::now();
    for _ in 0..iters {
        idx = unsafe { *arr.get_unchecked(idx) };
    }
    let elapsed = s.elapsed().as_nanos() as f64;
    std::hint::black_box(idx);
    elapsed / iters as f64
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
        std::thread::sleep(std::time::Duration::from_secs(180));
        eprintln!("\n!!! 180s TIMEOUT !!!");
        std::process::exit(0);
    });

    let caps = acpu::probe::scan();
    eprintln!(
        "=== acpu CPU driver benchmark — {:?} ({}P+{}E) ===",
        caps.chip, caps.p_cores, caps.e_cores
    );

    // ╔═══════════════════════════════════════════════════════════════════╗
    // ║  PART 1 — HARDWARE CHARACTERIZATION                             ║
    // ╚═══════════════════════════════════════════════════════════════════╝

    // ── MEMORY LATENCY (pointer chasing) ─────────────────────────────────

    eprintln!("\n  MEMORY LATENCY (pointer chasing, random access)");
    eprintln!("  {:<18} {:>10} {:>10}", "level", "size", "latency");
    eprintln!("  {}", "─".repeat(40));
    for &(bytes, label) in &[
        (16 * 1024, "L1  16KB"),
        (128 * 1024, "L2  128KB"),
        (1024 * 1024, "L2  1MB"),
        (4 * 1024 * 1024, "L3  4MB"),
        (32 * 1024 * 1024, "L3  32MB"),
        (128 * 1024 * 1024, "DRAM 128MB"),
    ] {
        let lat = chase_latency_ns(bytes);
        eprintln!("  {:<18} {:>8}KB {:>8.1}ns", label, bytes / 1024, lat);
    }

    // ── STREAM BANDWIDTH ─────────────────────────────────────────────────

    eprintln!("\n  STREAM BANDWIDTH (4M f32 = 16MB per array)");
    eprintln!("  {:<18} {:>10} {:>10}", "kernel", "GB/s", "bytes/op");
    eprintln!("  {}", "─".repeat(40));
    {
        let sn = 4 * 1024 * 1024usize; // 4M elements = 16MB
        let a_arr: Vec<f32> = (0..sn).map(|i| (i % 1000) as f32 * 0.001).collect();
        let b_arr: Vec<f32> = (0..sn).map(|i| (i % 997) as f32 * 0.001).collect();
        let mut c_arr = vec![0f32; sn];
        let mut d_arr = vec![0f32; sn];
        let scalar = 3.14159f32;
        let bytes = sn as f64 * 4.0;

        // copy: c = a  (read a + write c = 2×bytes)
        let t = ns(|| {
            c_arr.copy_from_slice(&a_arr);
            std::hint::black_box(&c_arr);
        });
        eprintln!(
            "  {:<18} {:>9.1} {:>10}",
            "copy",
            2.0 * bytes / t as f64,
            "2×N"
        );

        // scale: c = s*a  (read a + write c = 2×bytes)
        let t = ns(|| unsafe {
            use core::arch::aarch64::*;
            let sv = vdupq_n_f32(scalar);
            let pa = a_arr.as_ptr();
            let pc = c_arr.as_mut_ptr();
            let mut i = 0;
            while i + 16 <= sn {
                let a0 = vld1q_f32(pa.add(i));
                let a1 = vld1q_f32(pa.add(i + 4));
                let a2 = vld1q_f32(pa.add(i + 8));
                let a3 = vld1q_f32(pa.add(i + 12));
                vst1q_f32(pc.add(i), vmulq_f32(sv, a0));
                vst1q_f32(pc.add(i + 4), vmulq_f32(sv, a1));
                vst1q_f32(pc.add(i + 8), vmulq_f32(sv, a2));
                vst1q_f32(pc.add(i + 12), vmulq_f32(sv, a3));
                i += 16;
            }
            std::hint::black_box(&c_arr);
        });
        eprintln!(
            "  {:<18} {:>9.1} {:>10}",
            "scale",
            2.0 * bytes / t as f64,
            "2×N"
        );

        // add: c = a+b  (read a,b + write c = 3×bytes)
        let t = ns(|| unsafe {
            use core::arch::aarch64::*;
            let pa = a_arr.as_ptr();
            let pb = b_arr.as_ptr();
            let pc = c_arr.as_mut_ptr();
            let mut i = 0;
            while i + 16 <= sn {
                let a0 = vld1q_f32(pa.add(i));
                let a1 = vld1q_f32(pa.add(i + 4));
                let a2 = vld1q_f32(pa.add(i + 8));
                let a3 = vld1q_f32(pa.add(i + 12));
                let b0 = vld1q_f32(pb.add(i));
                let b1 = vld1q_f32(pb.add(i + 4));
                let b2 = vld1q_f32(pb.add(i + 8));
                let b3 = vld1q_f32(pb.add(i + 12));
                vst1q_f32(pc.add(i), vaddq_f32(a0, b0));
                vst1q_f32(pc.add(i + 4), vaddq_f32(a1, b1));
                vst1q_f32(pc.add(i + 8), vaddq_f32(a2, b2));
                vst1q_f32(pc.add(i + 12), vaddq_f32(a3, b3));
                i += 16;
            }
            std::hint::black_box(&c_arr);
        });
        eprintln!(
            "  {:<18} {:>9.1} {:>10}",
            "add",
            3.0 * bytes / t as f64,
            "3×N"
        );

        // triad: d = a + s*b  (read a,b + write d = 3×bytes)
        let t = ns(|| unsafe {
            use core::arch::aarch64::*;
            let sv = vdupq_n_f32(scalar);
            let pa = a_arr.as_ptr();
            let pb = b_arr.as_ptr();
            let pd = d_arr.as_mut_ptr();
            let mut i = 0;
            while i + 16 <= sn {
                let a0 = vld1q_f32(pa.add(i));
                let a1 = vld1q_f32(pa.add(i + 4));
                let a2 = vld1q_f32(pa.add(i + 8));
                let a3 = vld1q_f32(pa.add(i + 12));
                let b0 = vld1q_f32(pb.add(i));
                let b1 = vld1q_f32(pb.add(i + 4));
                let b2 = vld1q_f32(pb.add(i + 8));
                let b3 = vld1q_f32(pb.add(i + 12));
                vst1q_f32(pd.add(i), vfmaq_f32(a0, sv, b0));
                vst1q_f32(pd.add(i + 4), vfmaq_f32(a1, sv, b1));
                vst1q_f32(pd.add(i + 8), vfmaq_f32(a2, sv, b2));
                vst1q_f32(pd.add(i + 12), vfmaq_f32(a3, sv, b3));
                i += 16;
            }
            std::hint::black_box(&d_arr);
        });
        eprintln!(
            "  {:<18} {:>9.1} {:>10}",
            "triad",
            3.0 * bytes / t as f64,
            "3×N"
        );
    }

    // ── PREFETCH IMPACT ──────────────────────────────────────────────────
    // stride access defeats HW prefetcher — SW prefetch can help
    eprintln!("\n  PREFETCH IMPACT (stride access over 16MB)");
    eprintln!("  {:<18} {:>10} {:>10}", "mode", "GB/s", "speedup");
    eprintln!("  {}", "─".repeat(40));
    {
        let pn = 4 * 1024 * 1024usize;
        let pdata: Vec<f32> = vec![1.0; pn];
        let stride = 256usize; // 1KB stride — defeats simple HW prefetcher

        // strided sum without prefetch
        let t_no = ns(|| unsafe {
            let p = pdata.as_ptr();
            let mut acc = 0f32;
            let mut i = 0;
            while i < pn {
                acc += *p.add(i);
                i += stride;
            }
            std::hint::black_box(acc);
        });
        let elements_touched = pn / stride;
        let bytes_touched = elements_touched as f64 * 64.0; // each touches a cache line
        let bw_no = bytes_touched / t_no as f64;

        // strided sum with prefetch_l2 ahead
        let t_pf = ns(|| unsafe {
            let p = pdata.as_ptr();
            let mut acc = 0f32;
            let mut i = 0;
            let ahead = stride * 8; // prefetch 8 strides ahead
            while i + ahead < pn {
                acpu::sync::prefetch::prefetch_l2(p.add(i + ahead) as *const u8);
                acc += *p.add(i);
                i += stride;
            }
            while i < pn {
                acc += *p.add(i);
                i += stride;
            }
            std::hint::black_box(acc);
        });
        let bw_pf = bytes_touched / t_pf as f64;

        eprintln!("  {:<18} {:>9.1} {:>10}", "no prefetch", bw_no, "baseline");
        eprintln!(
            "  {:<18} {:>9.1} {:>8.2}×",
            "prefetch_l2",
            bw_pf,
            bw_pf / bw_no
        );
    }

    // ── P-CORE vs E-CORE ─────────────────────────────────────────────────

    eprintln!("\n  P-CORE vs E-CORE (same workload, different core class)");
    eprintln!(
        "  {:<18} {:>10} {:>10} {:>8}",
        "workload", "P-core", "E-core", "P/E"
    );
    eprintln!("  {}", "─".repeat(50));
    {
        // workloads defined with their own data inside
        // Use heavy workloads so QoS scheduling has time to place thread on correct core
        let workloads: Vec<(&str, Arc<dyn Fn() + Send + Sync>)> = vec![
            (
                "sum 1M",
                Arc::new({
                    let s: Vec<f32> = vec![1.0; 1024 * 1024];
                    move || {
                        std::hint::black_box(acpu::vector::reduce::sum(&s));
                    }
                }),
            ),
            (
                "exp 64K",
                Arc::new({
                    let s: Vec<f32> = (0..65536).map(|i| (i % 100) as f32 * 0.01).collect();
                    move || {
                        let mut b = s.clone();
                        acpu::vector::math::exp(&mut b);
                        std::hint::black_box(&b);
                    }
                }),
            ),
            (
                "sgemm 256",
                Arc::new({
                    let sz = 256;
                    let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.1).collect();
                    let b: Vec<f32> = (0..sz * sz).map(|i| (i % 11) as f32 * 0.1).collect();
                    move || {
                        let mut c = vec![0f32; sz * sz];
                        acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);
                        std::hint::black_box(&c);
                    }
                }),
            ),
        ];

        for (name, work) in &workloads {
            let w1 = work.clone();
            let w2 = work.clone();
            let reps = 100;

            let p_ns = std::thread::spawn(move || {
                let _ = acpu::sync::affinity::pin_p_core();
                std::thread::sleep(std::time::Duration::from_millis(10));
                for _ in 0..30 {
                    w1();
                }
                let mut t = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let s = Instant::now();
                    w1();
                    t.push(s.elapsed().as_nanos() as u64);
                }
                med(&mut t)
            })
            .join()
            .unwrap();

            let e_ns = std::thread::spawn(move || {
                let _ = acpu::sync::affinity::pin_e_core();
                std::thread::sleep(std::time::Duration::from_millis(10));
                for _ in 0..30 {
                    w2();
                }
                let mut t = Vec::with_capacity(reps);
                for _ in 0..reps {
                    let s = Instant::now();
                    w2();
                    t.push(s.elapsed().as_nanos() as u64);
                }
                med(&mut t)
            })
            .join()
            .unwrap();

            let ratio = p_ns as f64 / e_ns as f64;
            eprintln!("  {:<18} {:>9}ns {:>9}ns {:>6.2}×", name, p_ns, e_ns, ratio);
        }
    }

    // ── CROSS-CORE LATENCY ───────────────────────────────────────────────

    eprintln!("\n  CROSS-CORE LATENCY (atomic ping-pong, one-way)");
    eprintln!("  {:<18} {:>10}", "path", "latency");
    eprintln!("  {}", "─".repeat(30));
    {
        let rounds = 50_000u64;

        let ping_pong = |pin_a: fn(), pin_b: fn()| -> f64 {
            let flag = Arc::new(AtomicU64::new(0));
            let f2 = flag.clone();
            let handle = std::thread::spawn(move || {
                pin_b();
                for i in 0..rounds {
                    while f2.load(Ordering::Acquire) != i * 2 + 1 {
                        std::hint::spin_loop();
                    }
                    f2.store(i * 2 + 2, Ordering::Release);
                }
            });
            pin_a();
            std::thread::sleep(std::time::Duration::from_millis(1)); // let B start
            let start = Instant::now();
            for i in 0..rounds {
                flag.store(i * 2 + 1, Ordering::Release);
                while flag.load(Ordering::Acquire) != i * 2 + 2 {
                    std::hint::spin_loop();
                }
            }
            let elapsed = start.elapsed().as_nanos() as f64;
            handle.join().unwrap();
            elapsed / rounds as f64 / 2.0
        };

        fn pin_p() {
            let _ = acpu::sync::affinity::pin_p_core();
        }
        fn pin_e() {
            let _ = acpu::sync::affinity::pin_e_core();
        }

        let pp = ping_pong(pin_p, pin_p);
        let pe = ping_pong(pin_p, pin_e);
        eprintln!("  {:<18} {:>8.1}ns", "P ↔ P", pp);
        eprintln!("  {:<18} {:>8.1}ns", "P ↔ E", pe);
    }

    // ╔═══════════════════════════════════════════════════════════════════╗
    // ║  PART 2 — acpu vs APPLE ACCELERATE                              ║
    // ╚═══════════════════════════════════════════════════════════════════╝

    let n = 4096usize;
    let nn = n as i32;
    let nu = n as u64;
    let src: Vec<f32> = (0..n).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
    let pos: Vec<f32> = src.iter().map(|x| x.abs() + 0.01).collect();
    let mut b = src.clone();
    let mut d = vec![0f32; n];
    let mut d2 = vec![0f32; n];

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
    row(
        "gelu",
        ns(|| {
            b.copy_from_slice(&src);
            acpu::vector::math::gelu(&mut b);
        }),
        ns(|| unsafe {
            vDSP_vmul(src.as_ptr(), 1, src.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
            vDSP_vmul(d.as_ptr(), 1, src.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
            let c1: f32 = 0.044715;
            vDSP_vsmul(d.as_ptr(), 1, &c1, d.as_mut_ptr(), 1, nu);
            vDSP_vadd(src.as_ptr(), 1, d.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
            let c2: f32 = 0.7978845608;
            vDSP_vsmul(d.as_ptr(), 1, &c2, d.as_mut_ptr(), 1, nu);
            vvtanhf(d.as_mut_ptr(), d.as_ptr(), &nn);
            let one: f32 = 1.0;
            vDSP_vsadd(d.as_ptr(), 1, &one, d.as_mut_ptr(), 1, nu);
            let half: f32 = 0.5;
            vDSP_vsmul(src.as_ptr(), 1, &half, d2.as_mut_ptr(), 1, nu);
            vDSP_vmul(d2.as_ptr(), 1, d.as_ptr(), 1, d.as_mut_ptr(), 1, nu);
        }),
    );
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
        "length",
        ns(|| {
            std::hint::black_box(acpu::vector::reduce::length(&src));
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

    // ── COMPOUND ─────────────────────────────────────────────────────────

    hdr("COMPOUND (4096 f32)");
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
    let w: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.1).collect();
    let mut rm = vec![0f32; n];
    acpu::vector::softmax::normalize(&mut rm, &src, &w, 1e-5);
    row(
        "normalize",
        ns(|| {
            acpu::vector::softmax::normalize(&mut rm, &src, &w, 1e-5);
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

    // ── NUMERIC EXTENSIONS ───────────────────────────────────────────────

    eprintln!("\n  NUMERIC EXTENSIONS (4096 elements)");
    eprintln!("  {:<18} {:>10} {:>10}", "operation", "time", "throughput");
    eprintln!("  {}", "─".repeat(40));
    {
        let ne = 4096usize;
        let ne_bytes = ne as f64 * 4.0;

        // complex mul-acc (FCMA): interleaved [re,im,re,im,...] = 2048 complex pairs
        let ca: Vec<f32> = (0..ne).map(|i| (i % 17) as f32 * 0.1).collect();
        let cb: Vec<f32> = (0..ne).map(|i| (i % 13) as f32 * 0.1).collect();
        let mut cacc = vec![0f32; ne];
        let t = ns(|| {
            cacc.fill(0.0);
            acpu::numeric::complex::complex_mul_acc(&mut cacc, &ca, &cb);
            std::hint::black_box(&cacc);
        });
        eprintln!(
            "  {:<18} {:>9}ns {:>8.1} Ge/s",
            "complex_mul_acc",
            t,
            ne as f64 / t as f64
        );

        // bf16 round-trip: f32→bf16→f32
        let fsrc: Vec<f32> = (0..ne).map(|i| i as f32 * 0.01).collect();
        let mut bf16buf = vec![0u16; ne];
        let mut fout = vec![0f32; ne];
        let t = ns(|| {
            acpu::cast_f32_bf16(&mut bf16buf, &fsrc);
            acpu::cast_bf16_f32(&mut fout, &bf16buf);
            std::hint::black_box(&fout);
        });
        eprintln!(
            "  {:<18} {:>9}ns {:>8.1} GB/s",
            "bf16 round-trip",
            t,
            2.0 * ne_bytes / t as f64
        );

        // i8 quant round-trip: f32→i8→f32
        let mut i8buf = vec![0i8; ne];
        let t = ns(|| {
            acpu::cast_f32_i8(&mut i8buf, &fsrc, 0.01);
            acpu::cast_i8_f32(&mut fout, &i8buf, 0.01, 0);
            std::hint::black_box(&fout);
        });
        eprintln!(
            "  {:<18} {:>9}ns {:>8.1} GB/s",
            "i8 quant round-trip",
            t,
            2.0 * ne_bytes / t as f64
        );
    }

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
            acpu::cast_f32_f16(&mut f16b, &fd);
            std::hint::black_box(&f16b);
        }),
    );
    acpu::cast_f32_f16(&mut f16b, &fd);
    cvt_row(
        "f16→f32",
        ns(|| {
            acpu::cast_f16_f32(&mut fo, &f16b);
            std::hint::black_box(&fo);
        }),
    );
    cvt_row(
        "f32→bf16",
        ns(|| {
            acpu::cast_f32_bf16(&mut bfb, &fd);
            std::hint::black_box(&bfb);
        }),
    );
    acpu::cast_f32_bf16(&mut bfb, &fd);
    cvt_row(
        "bf16→f32",
        ns(|| {
            acpu::cast_bf16_f32(&mut fo, &bfb);
            std::hint::black_box(&fo);
        }),
    );
    cvt_row(
        "f32→i8",
        ns(|| {
            acpu::cast_f32_i8(&mut i8b, &fd, 0.1);
            std::hint::black_box(&i8b);
        }),
    );
    acpu::cast_f32_i8(&mut i8b, &fd, 0.1);
    cvt_row(
        "i8→f32",
        ns(|| {
            acpu::cast_i8_f32(&mut fo, &i8b, 0.1, 0);
            std::hint::black_box(&fo);
        }),
    );
    eprintln!("  (memcpy {} f32 = {}ns baseline)", n, memcpy_ns);

    // ── SGEMM ────────────────────────────────────────────────────────────

    let gemm_sizes: Vec<usize> = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    eprintln!("\n  SGEMM — full spectrum (GFLOPS, higher = better)");
    eprintln!(
        "  {:<18} {:>9} {:>9} {:>8}",
        "size", "acpu", "apple", "ratio"
    );
    eprintln!("  {}", "─".repeat(48));

    let gsz = gemm_sizes.clone();
    let apple_gemm = std::thread::spawn(move || {
        let mut results = Vec::new();
        for &sz in &gsz {
            let a = vec![1f32; sz * sz];
            let b = vec![1f32; sz * sz];
            let mut c_buf = vec![0f32; sz * sz];
            let it = if sz >= 2048 {
                5
            } else if sz >= 512 {
                10
            } else {
                30
            };
            // warmup
            for _ in 0..3 {
                c_buf.fill(0.0);
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
            }
            let mut best = u64::MAX;
            for _ in 0..it {
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
                let t = s.elapsed().as_nanos() as u64;
                if t < best {
                    best = t;
                }
            }
            results.push((sz, best));
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
            5
        } else if sz >= 512 {
            10
        } else {
            30
        };
        // warmup: 3 calls to warm caches and stabilize thermal
        for _ in 0..3 {
            c_buf.fill(0.0);
            acpu::matmul_f32(&a, &bm, &mut c_buf, sz, sz, sz);
        }
        let mut best = u64::MAX;
        for _ in 0..it {
            c_buf.fill(0.0);
            let s = Instant::now();
            acpu::matmul_f32(&a, &bm, &mut c_buf, sz, sz, sz);
            let t = s.elapsed().as_nanos() as u64;
            if t < best {
                best = t;
            }
        }
        let ac = apple_gemm.iter().find(|r| r.0 == sz).unwrap().1;
        let ops = 2.0 * (sz as f64).powi(3);
        row_gf(
            &format!("sgemm {sz}×{sz}"),
            ops / best as f64,
            ops / ac as f64,
        );
    }

    // ── HGEMM / BGEMM / QGEMM (reference, scalar) ────────────────────

    {
        eprintln!("\n  MIXED-PRECISION GEMM (256×256, reference impl)");
        eprintln!("  {:<18} {:>9} {:>9}", "operation", "GFLOPS", "note");
        eprintln!("  {}", "─".repeat(40));

        let sz = 256;
        let ops = 2.0 * (sz as f64).powi(3);

        // hgemm (fp16 in, fp32 accum)
        {
            let a16: Vec<u16> = (0..sz * sz)
                .map(|i| acpu::numeric::fp16::f32_to_fp16((i % 7) as f32 * 0.1))
                .collect();
            let b16: Vec<u16> = (0..sz * sz)
                .map(|i| acpu::numeric::fp16::f32_to_fp16((i % 11) as f32 * 0.1))
                .collect();
            let mut c = vec![0f32; sz * sz];
            acpu::matmul_f16(&a16, &b16, &mut c, sz, sz, sz); // warmup
            let mut best = u64::MAX;
            for _ in 0..5 {
                c.fill(0.0);
                let s = Instant::now();
                acpu::matmul_f16(&a16, &b16, &mut c, sz, sz, sz);
                let t = s.elapsed().as_nanos() as u64;
                if t < best {
                    best = t;
                }
            }
            let gf = ops / best as f64;
            eprintln!(
                "  {:<18} {:>7.1}GF  cvt+sgemm (FMA16 GEBP planned)",
                "hgemm fp16", gf
            );
        }

        // bgemm (bf16 in, fp32 accum)
        {
            let a16: Vec<u16> = (0..sz * sz)
                .map(|i| acpu::numeric::bf16::f32_to_bf16((i % 7) as f32 * 0.1))
                .collect();
            let b16: Vec<u16> = (0..sz * sz)
                .map(|i| acpu::numeric::bf16::f32_to_bf16((i % 11) as f32 * 0.1))
                .collect();
            let mut c = vec![0f32; sz * sz];
            acpu::matmul_bf16(&a16, &b16, &mut c, sz, sz, sz);
            let mut best = u64::MAX;
            for _ in 0..5 {
                c.fill(0.0);
                let s = Instant::now();
                acpu::matmul_bf16(&a16, &b16, &mut c, sz, sz, sz);
                let t = s.elapsed().as_nanos() as u64;
                if t < best {
                    best = t;
                }
            }
            let gf = ops / best as f64;
            eprintln!("  {:<18} {:>7.1}GF  cvt+sgemm", "bgemm bf16", gf);
        }

        // qgemm (i8 in, fp32 accum)
        {
            let a8: Vec<i8> = (0..sz * sz).map(|i| (i % 127) as i8).collect();
            let b8: Vec<i8> = (0..sz * sz).map(|i| (i % 127) as i8).collect();
            let mut c = vec![0f32; sz * sz];
            acpu::matmul_i8(&a8, &b8, &mut c, sz, sz, sz, 0.01, 0);
            let mut best = u64::MAX;
            for _ in 0..5 {
                c.fill(0.0);
                let s = Instant::now();
                acpu::matmul_i8(&a8, &b8, &mut c, sz, sz, sz, 0.01, 0);
                let t = s.elapsed().as_nanos() as u64;
                if t < best {
                    best = t;
                }
            }
            let gf = ops / best as f64;
            eprintln!(
                "  {:<18} {:>7.1}GF  dequant+sgemm (MAC16 planned)",
                "qgemm i8", gf
            );
        }
    }

    // ── ROPE ────────────────────────────────────────────────────────────

    {
        eprintln!("\n  ROPE (rotary positional embedding, 4096 dim)");
        let dim = 4096;
        let x: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let freqs: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let mut out = vec![0f32; dim];
        let rope_ns = ns(|| acpu::vector::rotate(&mut out, &x, &freqs, 42));
        eprintln!("  rotate {dim}: {}ns", rope_ns);
    }

    // ── THREAD SCALING (sgemm 1024×1024) ────────────────────────────────

    {
        eprintln!("\n  THREAD SCALING (sgemm 1024×1024)");
        let sz = 1024;
        let a: Vec<f32> = (0..sz * sz).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..sz * sz).map(|i| (i % 11) as f32 * 0.1).collect();
        let mut c = vec![0f32; sz * sz];
        let ops = 2.0 * (sz as f64).powi(3);
        // warmup
        for _ in 0..3 {
            c.fill(0.0);
            acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);
        }
        let mut best = u64::MAX;
        for _ in 0..10 {
            c.fill(0.0);
            let s = Instant::now();
            acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);
            let t = s.elapsed().as_nanos() as u64;
            if t < best {
                best = t;
            }
        }
        let gf = ops / best as f64;
        let cores = acpu::probe::scan().p_cores;
        eprintln!(
            "  {} P-cores: {:.0} GFLOPS ({:.0} GF/core)",
            cores,
            gf,
            gf / cores as f64
        );
    }

    // ╔═══════════════════════════════════════════════════════════════════╗
    // ║  PART 3 — HARDWARE PROFILING                                     ║
    // ╚═══════════════════════════════════════════════════════════════════╝

    // ── AMX UTILIZATION ──────────────────────────────────────────────────

    eprintln!("\n  AMX UTILIZATION (theoretical peak @ 3.228 GHz)");
    {
        let ta = [1f32; 256];
        let tb = [1f32; 256];
        let mut tc = [0f32; 256];
        acpu::matmul_f32(&ta, &tb, &mut tc, 16, 16, 16);
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

    // ── PMU COUNTERS (optional, requires root) ───────────────────────────

    eprintln!("\n  PMU COUNTERS (IPC + cache behavior)");
    match acpu::Counters::new(&[
        acpu::pulse::Counter::Cycles,
        acpu::pulse::Counter::Instructions,
        acpu::pulse::Counter::L1dMisses,
    ]) {
        Err(_) => {
            eprintln!("  (skipped — requires root or dtrace_proc entitlement)");
        }
        Ok(mut pmu) => {
            let pmu_src: Vec<f32> = (0..4096).map(|i| (i % 100) as f32 * 0.01).collect();
            let pmu_buf = pmu_src.clone();

            let pmu_workloads: Vec<(&str, Box<dyn FnMut()>)> = vec![
                (
                    "sum 4096",
                    Box::new({
                        let s = pmu_src.clone();
                        move || {
                            std::hint::black_box(acpu::vector::reduce::sum(&s));
                        }
                    }),
                ),
                (
                    "exp 4096",
                    Box::new({
                        let s = pmu_src.clone();
                        let mut b = pmu_buf.clone();
                        move || {
                            b.copy_from_slice(&s);
                            acpu::vector::math::exp(&mut b);
                        }
                    }),
                ),
                (
                    "sgemm 64",
                    Box::new(|| {
                        let sz = 64;
                        let a = vec![1.0f32; sz * sz];
                        let b = vec![1.0f32; sz * sz];
                        let mut c = vec![0f32; sz * sz];
                        acpu::matmul_f32(&a, &b, &mut c, sz, sz, sz);
                        std::hint::black_box(&c);
                    }),
                ),
            ];
            for (name, mut work) in pmu_workloads {
                work();
                pmu.start();
                let snap_a = pmu.read();
                for _ in 0..100 {
                    work();
                }
                let snap_b = pmu.read();
                pmu.stop();
                let counts = pmu.elapsed(&snap_a, &snap_b);
                let ipc = counts.instructions as f64 / counts.cycles.max(1) as f64;
                eprintln!(
                    "  {:<18} IPC={:.2}  cycles={}  L1d_miss={}",
                    name, ipc, counts.cycles, counts.l1d_misses
                );
            }
        }
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
