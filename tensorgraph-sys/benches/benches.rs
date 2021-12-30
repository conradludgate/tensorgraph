use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cust::quick_init;
use tensorgraph_sys::{
    device::{cpu::Cpu, Device},
    tensor::{gemm, Tensor},
    vec::Vec,
};

pub fn matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    let mut init = vec![0.0f64; 256 * 256];
    init[1] = 0.001;
    for i in 0..256 {
        let i = i * 256 + i; // diagonals
        init[i] = 1.0;
    }

    group.bench_function("openblas", |b| {
        b.iter(|| {
            let a = Vec::<f64, Cpu>::copy_from_host(&init);
            let b = a.clone();
            let c = b.clone();

            let mut a = Tensor::from_shape_in((), [256, 256], a); // 3 rows x 2 cols
            let b = Tensor::from_shape_in((), [256, 256], b); // 2 rows x 2 cols
            let mut c = Tensor::from_shape_in((), [256, 256], c); // 2 rows x 2 cols

            for _ in 0..1000 {
                gemm(1., a.view(), b.view(), 0., c.view_mut());
                std::mem::swap(&mut a, &mut c);
            }

            black_box(c.into_inner());
        })
    });

    {
        use tensorgraph_sys::device::cuda::Cuda;
        let ctx = quick_init().unwrap();

        let cuda = Cuda::new(ctx.get_unowned());
        let handle = cuda.init_cublas();

        group.bench_function("cublas", |b| {
            b.iter(|| {
                let a = Vec::copy_from_host_in(&init, cuda.clone());
                let b = a.clone();
                let c = b.clone();

                let mut a = Tensor::from_shape_in(handle, [256, 256], a); // 3 rows x 2 cols
                let b = Tensor::from_shape_in(handle, [256, 256], b); // 2 rows x 2 cols
                let mut c = Tensor::from_shape_in(handle, [256, 256], c); // 2 rows x 2 cols

                for _ in 0..1000 {
                    gemm(1., a.view(), b.view(), 0., c.view_mut());
                    std::mem::swap(&mut a, &mut c);
                }

                let mut out = vec![0.0f64; 256 * 256];
                Cuda::copy_to_host(&c.into_inner(), &mut out);

                black_box(out);
            })
        });

        cust::context::Context::drop(ctx).unwrap();
    }

    group.finish();
}

criterion_group!(benches, matmul);
criterion_main!(benches);
