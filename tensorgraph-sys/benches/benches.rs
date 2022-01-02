#![feature(allocator_api)]

use std::{alloc::Global, ops::Deref};

use criterion::{black_box, criterion_group, criterion_main, Bencher, Criterion};
use tensorgraph_sys::{
    blas::{cpu::CpuContext, BLASContext, GEMM},
    device::DeviceAllocator,
    tensor::{gemm, Tensor},
    vec::Vec,
};

/// Performs 1000 matrix mulitplications on a 256x256 matrix
pub fn matmul_1000_256<A: DeviceAllocator + Clone, C: BLASContext<Device = A::Device> + Copy>(
    init: &[f64],
    alloc: A,
    ctx: C,
) -> Vec<f64, A>
where
    f64: GEMM<C>,
{
    let a = Vec::copy_from_host_in(init, alloc);
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
        b.iter(|| black_box(matmul_1000_256(&init, Global, CpuContext)));
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
    {
        use tensorgraph_sys::blas::cublas::CublasContext;
        use tensorgraph_sys::device::cuda::{Context, Stream};

        let _ctx = Context::quick_init().unwrap();
        let cuda = Stream::new().unwrap();
        let cuda = cuda.deref();
        let ctx = CublasContext::new();
        let ctx = cuda.init_cublas(&ctx);

        group.bench_function("cublas", |b| {
            b.iter(|| {
                // includes the time to sync data in the benchmark
                let mut out = vec![0.0f64; 256 * 256];
                matmul_1000_256(&init, cuda, ctx).copy_to_host(&mut out);

                black_box(out)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, matmul);
criterion_main!(benches);
