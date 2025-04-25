use plotters::prelude::*;
use tch::{nn::{self, Module}, Device, Kind, Tensor};

struct SelfAttention {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    fc_out: nn::Linear,
    embedding_dim: i64,
    num_heads: i64,
    head_dim: i64,
}

impl SelfAttention {
    fn new(vs: &nn::Path, embedding_dim: i64, num_heads: i64) -> Self {
        let head_dim = embedding_dim / num_heads;
        assert_eq!(
            head_dim * num_heads,
            embedding_dim,
            "embedding_dim must be divisible by num_heads"
        );
        Self {
            query: nn::linear(vs, embedding_dim, embedding_dim, Default::default()),
            key: nn::linear(vs, embedding_dim, embedding_dim, Default::default()),
            value: nn::linear(vs, embedding_dim, embedding_dim, Default::default()),
            fc_out: nn::linear(vs, embedding_dim, embedding_dim, Default::default()),
            embedding_dim,
            num_heads,
            head_dim,
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let (batch_size, seq_length, _) = xs.size3().unwrap();
        let q = self
            .query
            .forward(xs)
            .view([batch_size, seq_length, self.num_heads, self.head_dim])
            .permute(&[0, 2, 1, 3]);
        let k = self
            .key
            .forward(xs)
            .view([batch_size, seq_length, self.num_heads, self.head_dim])
            .permute(&[0, 2, 1, 3]);
        let v = self
            .value
            .forward(xs)
            .view([batch_size, seq_length, self.num_heads, self.head_dim])
            .permute(&[0, 2, 1, 3]);
        // Attention scores
        let attn = q.matmul(&k.transpose(-2, -1)) / (self.head_dim as f64).sqrt();
        let attn_weights = attn.softmax(-1, Kind::Float);
        // Apply attention
        let out = attn_weights
            .matmul(&v)
            .permute(&[0, 2, 1, 3])
            .contiguous()
            .view([batch_size, seq_length, self.embedding_dim]);
        self.fc_out.forward(&out)
    }
}

fn plot_attention(attn_matrix: &Tensor, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<f32> = attn_matrix.reshape(&[-1]).try_into()?;
    let size = attn_matrix.size();
    let (rows, cols) = (size[0] as usize, size[1] as usize);
    let root = BitMapBackend::new(filename, (480, 400)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Self-Attention Weights", ("sans", 30))
        .margin(20)
        .build_cartesian_2d(0..cols, 0..rows)?;
    chart
        .configure_mesh()
        .x_desc("Key position")
        .y_desc("Query position")
        .disable_mesh()
        .draw()?;
    for i in 0..rows {
        for j in 0..cols {
            let v = data[i * cols + j];
            let color = HSLColor(0.7 - v as f64 * 0.7, 0.6, 0.5 + v as f64 * 0.5);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(j, i), (j + 1, i + 1)],
                color.filled(),
            )))?;
        }
    }
    root.present()?;
    Ok(())
}

fn visualize_attention() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available();
    let embedding_dim = 8;
    let num_heads = 1;
    let seq_length = 5;
    let batch_size = 1;
    let vs = nn::VarStore::new(device);
    let attention = SelfAttention::new(&vs.root(), embedding_dim, num_heads);
    let x = Tensor::randn(&[batch_size, seq_length, embedding_dim], (Kind::Float, device));
    let output = attention.forward(&x);
    // 计算注意力权重
    let q = attention.query.forward(&x)
        .view([batch_size, seq_length, 1, embedding_dim])
        .permute(&[0, 2, 1, 3]);
    let k = attention.key.forward(&x)
        .view([batch_size, seq_length, 1, embedding_dim])
        .permute(&[0, 2, 1, 3]);
    let attn_weights = (q.matmul(&k.transpose(-1, -2)) / (embedding_dim as f64).sqrt()).softmax(-1, Kind::Float);
    let attn_matrix = attn_weights.get(0).get(0); // shape: (seq_length, seq_length)
    plot_attention(&attn_matrix, "attention_weights.png")?;
    println!("Input shape: {:?}", x.size());
    println!("Output shape: {:?}", output.size());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    visualize_attention()
}
