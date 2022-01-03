use criterion::{black_box, criterion_group, criterion_main, Bencher, Criterion};
use tensorgraph_sys::{
    device::DefaultDeviceAllocator,
    vec::{vec_from_host, Vec},
};

use tensorgraph_math::{
    blas::{cpu::CpuContext, BLASContext, GEMM},
    tensor::{gemm, Tensor},
};

/// Performs 1000 matrix mulitplications on a 256x256 matrix
pub fn matmul_1000_256<D: DefaultDeviceAllocator, C: BLASContext<Device = D> + Copy>(
    init: &[f64],
    ctx: C,
) -> Vec<f64, D::Alloc>
where
    f64: GEMM<C>,
    D::Alloc: Clone,
{
    let a = vec_from_host::<f64, D>(init);
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
        b.iter(|| black_box(matmul_1000_256(&init, CpuContext)));
    };

    #[cfg(feature = "openblas")]
    group.bench_function("openblas", cpu);

    #[cfg(feature = "blis")]
    group.bench_function("blis", cpu);

    #[cfg(feature = "netlib")]
    group.bench_function("netlib", cpu);

    #[cfg(feature = "matrixmultiply")]
    group.bench_function("matrixmultiply", cpu);

    #[cfg(feature = "accelerate")]
    group.bench_function("accelerate", cpu);

    #[cfg(feature = "cublas")]
    {
        use tensorgraph_math::blas::cublas::CublasContext;
        use tensorgraph_sys::device::cuda::{with_stream, Context, Stream};

        let _ctx = Context::quick_init().unwrap();

        with_stream(&Stream::new().unwrap(), |cuda| {
            let ctx = CublasContext::new();
            let ctx = ctx.with_stream(Some(cuda));

            group.bench_function("cublas", |b| {
                b.iter(|| {
                    // includes the time to sync data in the benchmark
                    let mut out = vec![0.0f64; 256 * 256];
                    matmul_1000_256(&init, ctx).copy_to_host(&mut out);

                    black_box(out)
                });
            });
        });
    }

    group.finish();
}

criterion_group!(benches, matmul);
criterion_main!(benches);
