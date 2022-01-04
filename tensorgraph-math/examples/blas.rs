use tensorgraph_math::tensor::Tensor;
use tensorgraph_math::sys::View;

fn main() {
    //     0 1
    // A = 2 3
    //     4 5

    // B = 0 1
    //     2 3

    // column major (read each column first)
    let a = [0., 2., 4., 1., 3., 5.];
    let b = [0., 2., 1., 3.];

    let a = Tensor::from_shape([3, 2], a); // 3 rows x 2 cols
    let b = Tensor::from_shape([2, 2], b); // 2 rows x 2 cols

    //           2  3
    // C = AB =  6 11
    //          10 19

    let c = a.dot(b.view());
    assert_eq!(c.into_inner().into_std(), [2., 6., 10., 3., 11., 19.]);
}
