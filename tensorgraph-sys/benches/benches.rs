use criterion::{black_box, criterion_group, criterion_main, Bencher, Criterion};
use tensorgraph_sys::{
    blas::{BLASDevice, GEMM},
    device::{cpu::Cpu, Device},
    tensor::{gemm, Tensor},
    vec::Vec,
};

/// Performs 1000 matrix mulitplications on a 256x256 matrix
pub fn matmul_1000_256<D: BLASDevice + Clone>(
    init: &[f64],
    device: D,
    ctx: D::Context,
) -> Vec<f64, D>
where
    D::Context: Copy,
    f64: GEMM<D>,
{
    let a = Vec::copy_from_host_in(init, device);
    let b = a.clone();
    let c = b.clone();

    let mut a = Tensor::from_shape_in(ctx, [256, 256], a);
    let b = Tensor::from_shape_in(ctx, [256, 256], b);
    let mut c = Tensor::from_shape_in(ctx, [256, 256], c);

    for _ in 0..1000 {
        gemm(1., a.view(), b.view(), 0., c.view_mut());
        std::mem::swap(&mut a, &mut c);
    }

    c.into_inner()
}

pub fn matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    let mut init = vec![0.0f64; 256 * 256];
    init[1] = 0.001;
    for i in 0..256 {
        let i = i * 256 + i; // diagonals
        init[i] = 1.0;
    }

    let cpu = |b: &mut Bencher| {
        b.iter(|| black_box(matmul_1000_256(&init, Cpu::default(), ())));
    };

    #[cfg(feature = "openblas")]
    group.bench_function("openblas", cpu);

    #[cfg(feature = "blis")]
    group.bench_function("blis", cpu);

    #[cfg(feature = "netlib")]
    group.bench_function("netlib", cpu);

    #[cfg(feature = "matrixmultiply")]
    group.bench_function("matrixmultiply", cpu);

    #[cfg(feature = "cublas")]
    group.bench_function("cublas", |b| {
        use tensorgraph_sys::device::cuda::{Context, Cuda, Stream};

        // setup device and cublas contexts
        let cuda_ctx = Context::quick_init().unwrap();
        let cuda_stream = Stream::new().unwrap();
        let cuda = Cuda::new(cuda_ctx.share(), cuda_stream.share());
        let ctx = cuda.init_cublas();

        b.iter(|| {
            // includes the time to sync data in the benchmark
            let mut out = vec![0.0f64; 256 * 256];
            Cuda::copy_to_host(&matmul_1000_256(&init, cuda, ctx), &mut out);

            black_box(out)
        });
    });

    group.finish();
}

criterion_group!(benches, matmul);
criterion_main!(benches);
