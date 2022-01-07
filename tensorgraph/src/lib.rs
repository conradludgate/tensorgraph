use graph::{TensorInternal, TID};
pub use graph::{Context, ContextMut, Graph, GraphLike};
use smallvec::SmallVec;
use tensorgraph_math::{sys::device::Device, tensor};

mod graph;
mod ops;

type TensorViewMut<'a, F, D> = tensor::TensorViewMut<'a, F, D, Dim>;

/// Vec fits into 3 usizes, but 4 dimensions are fairly common in convolusions
/// so we're taking the hit. But it's not that important, since most of the time we won't be using vecs
type Dim = SmallVec<[usize; 4]>;
// type Name = SmallString<[u8; 32]>;

pub struct Tensor<'graph, F, D: Device> {
    id: TID,
    graph: &'graph Graph<F, D>,
}

impl<'graph, F, D: Device> Tensor<'graph, F, D> {
    fn shape(&self) -> Dim {
        // let tensors = self.graph.tensors.borrow();
        // let slice: &[TensorInternal<F, D>] = tensors.as_ref();
        // let inner = &slice[self.id.0];
        // inner.shape.clone()
        todo!()
    }
}
