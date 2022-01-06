use std::cell::RefCell;

use eyre::{Result, ensure};
use smallstr::SmallString;
use smallvec::SmallVec;
use tensorgraph_math::{
    sys::{
        device::{Device, DeviceAllocator},
        ptr::Ref,
        Vec,
    },
    tensor::TensorViewMut,
};

pub struct Graph<F, D: Device> {
    pub(crate) tensors: RefCell<Vec<TensorInternal<F, D>>>,
}

pub struct GraphBuffer<'graph, F, A: DeviceAllocator> {
    graph: &'graph Graph<F, A::Device>,
    memory: Vec<F, A>,
}

pub struct GraphRef<'graph, F, D: Device> {
    graph: &'graph Graph<F, D>,
    memory: Ref<[F], D>,
}

impl<F, D: Device> Graph<F, D> {
    pub fn apply(
        &self,
        op: impl Op<F, D> + 'static,
        inputs: &[Tensor<F, D>],
    ) -> Result<Tensor<F, D>> {
        let tensors = self.tensors.borrow();
        let id = tensors.len();

        let input_shapes: InputArray<Dim> = inputs.iter().map(Tensor::shape).collect();
        let output_shape = op.output_shape(&input_shapes)?;

        drop(tensors);

        let mut tensors = self.tensors.borrow_mut();

        tensors.push(TensorInternal {
            shape: output_shape,
            op: Box::new(op),
            inputs: inputs.iter().map(|t| t.id).collect(),
        });

        Ok(Tensor { id, graph: self })
    }
}

pub struct TensorInternal<F, D: Device> {
    shape: Dim,
    op: Box<dyn Op<F, D>>,
    inputs: InputArray<usize>,
}

pub type InputArray<T> = SmallVec<[T; 2]>;

pub trait Op<F, D: Device> {
    /// Validate the given input shapes and determines the output shape.
    /// (This is used to pre-allocate your output buffer)
    fn output_shape(&self, inputs: &[Dim]) -> Result<Dim>;

    fn compute(&self, ctx: &mut ComputeContext<'_, F, D>);
    fn grad(&self, ctx: &mut GradiantContext<'_, F, D>);

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

struct Variable;
impl<F, D: Device> Op<F, D> for Variable {
    fn output_shape(&self, _inputs: &[Dim]) -> Result<Dim> {
        Err(eyre::eyre!("Variable is just a marker trait and should not be used as an op. This is a bug in tensorgraph, please report: https://github.com/conradludgate/tensorgraph/issues"))
    }
    fn compute(&self, _ctx: &mut ComputeContext<'_, F, D>) {}
    fn grad(&self, _ctx: &mut GradiantContext<'_, F, D>) {}
}

struct AddOp;

impl<F, D: Device> Op<F, D> for AddOp {
    fn output_shape(&self, inputs: &[Dim]) -> Result<Dim> {
        match inputs {
            [lhs, rhs] => {
                if lhs == rhs {
                    Ok(lhs.clone())
                } else {
                    Err(eyre::eyre!("invalid shapes"))
                }
            }
            _ => Err(eyre::eyre!("add only accepts 2 inputs")),
        }
    }

    fn compute(&self, ctx: &mut ComputeContext<'_, F, D>) {
        
    }

    fn grad(&self, ctx: &mut GradiantContext<'_, F, D>) {
        todo!()
    }
}

pub struct ComputeContext<'graph, F, D: Device> {
    graph: &'graph mut GraphRef<'graph, F, D>,
    inputs: InputArray<Tensor<'graph, F, D>>,
    output: Tensor<'graph, F, D>,
}

pub struct GradiantContext<'graph, F, D: Device> {
    buf: &'graph Graph<F, D>,

    dinputs: InputArray<Tensor<'graph, F, D>>,
    doutput: Tensor<'graph, F, D>,
    output: Tensor<'graph, F, D>,
}

/// Vec fits into 3 usizes, but 4 dimensions are fairly common in convolusions
/// so we're taking the hit. But it's not that important, since most of the time we won't be using vecs
type Dim = SmallVec<[usize; 4]>;
// type Name = SmallString<[u8; 32]>;

pub struct Tensor<'graph, F, D: Device> {
    id: usize,
    graph: &'graph Graph<F, D>,
}

impl<'graph, F, D: Device> Tensor<'graph, F, D> {
    fn shape(&self) -> Dim {
        let tensors = self.graph.tensors.borrow();
        let slice: &[TensorInternal<F, D>] = tensors.as_ref();
        let inner = &slice[self.id];
        inner.shape.clone()
    }
}
