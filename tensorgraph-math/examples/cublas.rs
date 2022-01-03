use tensorgraph_sys::{
    device::cuda::{Context, Stream},
    DefaultVec, View,
};

use tensorgraph_math::{blas::cublas::CublasContext, tensor::Tensor};

fn main() {
    // init cuda context
    let cuda_ctx = Context::quick_init().unwrap();

    // create cuda stream and configure it as the global
    let stream = Stream::new(&cuda_ctx).unwrap();
    let _handle = stream.as_global();

    // create cublas context and configure it as the global
    let cublas_ctx = CublasContext::new();
    let _handle = cublas_ctx.with_stream(Some(&stream)).as_global();

    run()
}

fn run() {
    //     0 1
    // A = 2 3
    //     4 5

    // B = 0 1
    //     2 3

    // column major (read each column first)
    let a = DefaultVec::<f32, Cuda>::copy_from_host(&[0., 2., 4., 1., 3., 5.]);
    let b = DefaultVec::<f32, Cuda>::copy_from_host(&[0., 2., 1., 3.]);

    let a = Tensor::from_shape([3, 2], a); // 3 rows x 2 cols
    let b = Tensor::from_shape([2, 2], b); // 2 rows x 2 cols

    //           2  3
    // C = AB =  6 11
    //          10 19

    let c = a.dot(b.view());
    assert_eq!(c.into_inner().into_std(), [2., 6., 10., 3., 11., 19.]);
}
