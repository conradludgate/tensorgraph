use criterion::{black_box, criterion_group, criterion_main, Bencher, Criterion};
use tensorgraph_sys::{
    device::{cpu::Cpu, DefaultDeviceAllocator},
    DefaultVec, View, ViewMut,
};

use tensorgraph_math::{
    blas::{DefaultBLASContext, GEMM},
    tensor::{gemm_ctx, Tensor},
};

/// Performs 1000 matrix mulitplications on a 256x256 matrix
pub fn matmul_1000_256<D: DefaultDeviceAllocator + DefaultBLASContext>(
    init: &[f64],
) -> DefaultVec<f64, D>
where
    f64: GEMM<D::Context, D>,
    D::Alloc: Clone,
    D::Context: Copy,
{
    let a = DefaultVec::<f64, D>::copy_from_host(init);
    let b = a.clone();
    let c = b.clone();

    let mut a = Tensor::from_shape([256, 256], a);
    let b = Tensor::from_shape([256, 256], b);
    let mut c = Tensor::from_shape([256, 256], c);

    let ctx = D::Context::default();
    for _ in 0..1000 {
        gemm_ctx(ctx, 1., a.view(), b.view(), 0., c.view_mut());
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
        b.iter(|| black_box(matmul_1000_256::<Cpu>(&init)));
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
        use tensorgraph_sys::device::cuda::{Context, Cuda, CudaUnified, Stream};

        let cuda_ctx = Context::quick_init().unwrap();
        let stream = Stream::new(&cuda_ctx).unwrap();
        let _handle = stream.as_global();
        let cublas_ctx = CublasContext::new();
        let _handle = cublas_ctx.with_stream(Some(&stream)).as_global();

        group.bench_function("cublas", |b| {
            b.iter(|| {
                // includes the time to sync data in the benchmark
                let mut out = vec![0.0f64; 256 * 256];
                matmul_1000_256::<Cuda>(&init).copy_to_host(&mut out);

                black_box(out)
            });
        });

        group.bench_function("cublas_unified", |b| {
            b.iter(|| {
                // includes the time to sync data in the benchmark
                let mut out = vec![0.0f64; 256 * 256];
                matmul_1000_256::<CudaUnified>(&init).copy_to_host(&mut out);

                black_box(out)
            });
        });
    }

    #[cfg(feature = "cublas")]
    {
        use tensorgraph_math::blas::cublas::CublasContext;
        use tensorgraph_sys::device::cuda::{Context, CudaUnified};

        let _cuda_ctx = Context::quick_init().unwrap();
        let cublas_ctx = CublasContext::new();
        let _handle = cublas_ctx.with_stream(None).as_global();

        group.bench_function("cublas_unified_sync", |b| {
            b.iter(|| {
                // includes the time to sync data in the benchmark
                let mut out = vec![0.0f64; 256 * 256];
                matmul_1000_256::<CudaUnified>(&init).copy_to_host(&mut out);

                black_box(out)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, matmul);
criterion_main!(benches);
