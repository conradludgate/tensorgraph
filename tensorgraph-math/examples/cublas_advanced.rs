use std::{thread, time::Duration, ops::Deref};

use tensorgraph_sys::{
    device::cuda::{Context, SharedStream, Stream},
    Vec, View,
};

use tensorgraph_math::{
    blas::cublas::{CublasContext, SharedCublasContext},
    tensor::Tensor,
};

fn main() {
    // init cuda context
    let cuda_ctx = Context::quick_init().unwrap();

    // create cuda stream
    let stream = Stream::new(&cuda_ctx).unwrap();

    // create cublas context, with the provided stream
    let cublas_ctx = CublasContext::new();
    let cublas_ctx = cublas_ctx.with_stream(Some(&stream));

    run(cublas_ctx, &stream);
}

fn run(ctx: &SharedCublasContext, alloc: &SharedStream) {
    //     0 1
    // A = 2 3
    //     4 5

    // B = 0 1
    //     2 3

    // column major (read each column first)
    let a = Vec::copy_from_host_in(&[0., 2., 4., 1., 3., 5.0_f64], alloc);
    let b = Vec::copy_from_host_in(&[0., 2., 1., 3.], alloc);

    let a = Tensor::from_shape([3, 2], a); // 3 rows x 2 cols
    let b = Tensor::from_shape([2, 2], b); // 2 rows x 2 cols

    //           2  3
    // C = AB =  6 11
    //          10 19

    let c = a.dot_into(b.view(), ctx, alloc);

    let mut out = [0.; 6];
    c.into_inner().copy_to_host(&mut out);
    assert_eq!(out, [2., 6., 10., 3., 11., 19.]);
}
