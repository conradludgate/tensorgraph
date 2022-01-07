use tensorgraph as tg;
use tg::{Tensor, Graph, Feed, grads, zeroes};
// use tg::ops::{matmul}
// use tg::nn::{optimisers::Adam, activation::{relu, sigmoid}, initalisers::xavier};

fn main() {
    let g = Graph::default();

    // 0 implies that this dimension is unspecified
    // but will need to be specified when fed into the graph
    let input = g.placeholder([0, 28*28], "input");
    let exp = g.placeholder([0, 10], "expected");

    // create the graph to the output
    let mnist = g.ns("mnist_model");
    let output = model(mnist, input);

    println!("{:?}", mnist.summary());

    let params = mnist.current_vars();

    // compute the loss of the network
    let loss = mse(output, exp);

    // compute the grads of the loss against the network params
    let grads = grads(loss, params);

    // create the optimiser of the network params
    let adam = Adam::default(g.ns("adam"), params);

    // initialise the graph variables
    let mut ctx = g.init();

    // run the model for 20 epochs
    for epoch in 0..20 {
        // get inputs and expected values in batches of 256
        for (i, e) in batches(256, inputs, expected) {
            let feed = Feed::new()
                .with(input, i)
                .with(exp, e);

            // update the model
            adam.optimise(grads, feed, &mut ctx);
        }
    }

    // inference
    let test_output = ctx.eval(output, Feed::new().with(input, test_input));
}

fn model<'g>(g: impl GraphLike<'g>, x: Tensor<'g>) -> Tensor<'g> {
    let h0 = dense(g.ns("layer0"), input, 16);
    let a0 = relu(h0);

    let h1 = dense(g.ns("layer1"), a0, 16);
    let a1 = relu(h1);

    let h2 = dense(g.ns("layer2"), a1, 10);
    let a2 = sigmoid(h2);

    a2
}

fn dense<'g>(g: impl GraphLike<'g>, x: Tensor<'g>, output_size: usize) -> Tensor<'g> {
    let input_size = x.shape().last().unwrap();

    // define two variables for this dense layer
    // along with their initialisers
    let w = g.var("w").with_init(xavier([input_size, output_size]));
    let b = g.var("b").with_init(zeroes([output_size]));

    matmul(x, w) + b
}
