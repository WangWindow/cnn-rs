use tch::{
    Device, Kind, Tensor,
    nn::OptimizerConfig,
    nn::{self, Module},
};

// 定义一个简单的CNN结构
#[derive(Debug)]
struct SimpleCNN {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl SimpleCNN {
    fn new(vs: &nn::Path) -> Self {
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, 1024, 128, Default::default());
        let fc2 = nn::linear(vs, 128, 10, Default::default());
        Self {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

impl nn::Module for SimpleCNN {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.view([-1, 1, 28, 28])
            .apply(&self.conv1)
            .relu()
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .relu()
            .max_pool2d_default(2)
            .view([-1, 1024])
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
    }
}

fn train_and_test() -> Result<(), Box<dyn std::error::Error>> {
    let m = tch::vision::mnist::load_dir("data/MNIST/raw")?;
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let net = SimpleCNN::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..=3 {
        for (bimages, blabels) in m.train_iter(64).shuffle().to_device(device) {
            let logits = net.forward(&bimages);
            let loss = logits.cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        println!("epoch {epoch} done");
    }
    // 测试
    let test_accuracy = m
        .test_iter(256)
        .to_device(device)
        .map(|(bimages, blabels)| {
            let logits = net.forward(&bimages);
            let predicted = logits.argmax(-1, false);
            let correct = predicted
                .eq_tensor(&blabels)
                .to_kind(Kind::Float)
                .mean(Kind::Float)
                .double_value(&[]);
            correct
        })
        .sum::<f64>()
        / (m.test_images.size()[0] as f64 / 256.0);
    println!("Test accuracy: {:.2}%", test_accuracy * 100.0);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    train_and_test()
}
