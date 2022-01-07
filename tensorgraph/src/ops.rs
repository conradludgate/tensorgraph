use eyre::Result;
use smallvec::SmallVec;
use tensorgraph_math::sys::device::Device;

use crate::{
    graph::{ContextMut, Graph},
    Dim, Tensor,
};

pub type InputArray<T> = SmallVec<[T; 2]>;

pub trait Op<F, D: Device> {
    /// Validate the given input shapes and determines the output shape.
    /// (This is used to pre-allocate your output buffer)
    fn output_shape(&self, inputs: &[Dim]) -> Result<Dim>;

    fn compute(&self, ctx: ComputeContext<'_, '_, F, D>);
    fn grad(&self, ctx: GradiantContext<'_, F, D>);

    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

pub struct Variable;
impl<F, D: Device> Op<F, D> for Variable {
    fn output_shape(&self, _inputs: &[Dim]) -> Result<Dim> {
        Err(eyre::eyre!("Variable is just a marker trait and should not be used as an op. This is a bug in tensorgraph, please report: https://github.com/conradludgate/tensorgraph/issues"))
    }
    fn compute(&self, _ctx: ComputeContext<'_, '_, F, D>) {}
    fn grad(&self, _ctx: GradiantContext<'_, F, D>) {}
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

    fn compute(&self, ctx: ComputeContext<'_, '_, F, D>) {}

    fn grad(&self, ctx: GradiantContext<'_, F, D>) {
        todo!()
    }
}

pub struct ComputeContext<'ctx, 'graph, F, D: Device> {
    inputs: InputArray<Tensor<'graph, F, D>>,
    output: Tensor<'graph, F, D>,
    ctx: ContextMut<'ctx, 'graph, F, D>,
}

pub struct GradiantContext<'graph, F, D: Device> {
    buf: &'graph Graph<F, D>,

    dinputs: InputArray<Tensor<'graph, F, D>>,
    doutput: Tensor<'graph, F, D>,
    output: Tensor<'graph, F, D>,
}
