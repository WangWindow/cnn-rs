use ndarray::prelude::*;
use tch::Tensor;

fn main() {
    // 使用ndarray创建数组
    let arr = array![3, 1, 4, 1, 5];
    // 将ndarray数组转换为Vec
    let vec = arr.to_vec();
    // 用tch创建Tensor
    let t = Tensor::from_slice(&vec);
    // 张量乘以2
    let t = t * 2;
    t.print();
}
