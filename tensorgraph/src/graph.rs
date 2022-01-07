use std::{borrow::BorrowMut, cell::RefCell, collections::HashMap};

use eyre::Result;
use tensorgraph_math::sys::{
    device::{Device, DeviceAllocator},
    ptr::Ref,
    Vec,
};

use crate::{
    ops::{InputArray, Op, Variable},
    Dim, Tensor,
};

type StdVec<T> = std::vec::Vec<T>;
type NS = StdVec<&'static str>;
#[derive(Clone, Copy)]
pub struct TID(pub usize);

#[derive(Default)]
pub struct Graph<F, D: Device> {
    pub(crate) tensors: RefCell<StdVec<TensorInternal<F, D>>>,
    variables: RefCell<HashMap<NS, TID>>,
}

pub struct Context<'graph, F, A: DeviceAllocator> {
    graph: &'graph Graph<F, A::Device>,
    memory: Vec<F, A>,
}

impl<'graph, F, A: DeviceAllocator> Context<'graph, F, A> {
    fn as_mut(&mut self) -> ContextMut<'_, 'graph, F, A::Device> {
        let graph = self.graph;
        let memory = self.memory.as_mut();
        ContextMut { graph, memory }
    }
}

pub struct ContextMut<'ctx, 'graph, F, D: Device> {
    graph: &'graph Graph<F, D>,
    memory: &'ctx mut Ref<[F], D>,
}

impl<F, D: Device> Graph<F, D> {
    pub fn apply(&self, op: impl Op<F, D> + 'static, inputs: &[Tensor<F, D>]) -> Tensor<F, D> {
        let mut tensors = self.tensors.borrow_mut();
        let id = TID(tensors.len());

        tensors.push(TensorInternal {
            op: Box::new(op),
            inputs: inputs.iter().map(|t| t.id).collect(),
        });

        Tensor { id, graph: self }
    }
}

pub struct TensorInternal<F, D: Device> {
    op: Box<dyn Op<F, D>>,
    inputs: InputArray<TID>,
}

pub trait GraphLike<'graph, F, D: Device>: Sized {
    fn namespace(&self, name: &'static str) -> Namespace<'graph, F, D>;

    fn var(&self, name: &'static str) -> Tensor<'graph, F, D> {
        let ns = self.namespace(name);
        let tensor = ns.graph.apply(Variable, &[]);
        let mut vars = ns.graph.variables.borrow_mut();
        vars.insert(ns.namespace, tensor.id);
        tensor
    }
}

impl<'graph, F, D: Device> GraphLike<'graph, F, D> for &'graph Graph<F, D> {
    fn namespace(&self, name: &'static str) -> Namespace<'graph, F, D> {
        Namespace {
            graph: self,
            namespace: vec![name],
        }
    }
}

#[derive(Clone)]
pub struct Namespace<'graph, F, D: Device> {
    graph: &'graph Graph<F, D>,
    namespace: NS,
}

impl<'graph, F, D: Device> GraphLike<'graph, F, D> for Namespace<'graph, F, D> {
    fn namespace(&self, name: &'static str) -> Namespace<'graph, F, D> {
        let mut namespace = self.namespace.clone();
        namespace.push(name);
        Namespace {
            graph: self.graph,
            namespace,
        }
    }
}
