#[allow(dead_code)]
use rand::prelude::*;
use treerustler::data;
use treerustler::utils;

fn create_vector(size: usize, n_classes: u8) -> Vec<u8> {
    let mut rng: ThreadRng = rand::thread_rng();
    (0..size).map(|_| rng.gen_range(0..n_classes)).collect()
}

fn main() {
    let v: Vec<u8> = create_vector(10, 2);
    println!("{:?}", v);
    let entropy: f64 = utils::entropy_loss(&v);
    println!("Entropy Loss: {}", entropy);
    let gini: f64 = utils::gini_index(&v);
    println!("Gini Index: {}", gini);
    let data: data::Data = data::Data::from_random(10, 3);
    println!("{:#?}", data);

    let d: data::Data =
        data::Data::from_string(&"1.1 2.4; 1.1 0.0; 5.5 3.1; 4.0 4.0; 4.0 1.25; 6.1 3.1");
    let col_idx: usize = 1;
    let threshold: f64 = 2.0;
    let col_values: Vec<f64> = d.get_col(col_idx);
    let unique_values: Vec<f64> = d.unique(col_idx);
    let lt_eq: Vec<bool> = d.lt_eq(col_idx, threshold);

    println!("Data = {:#?}", d);
    println!("Data[:, {}] = {:?}", col_idx, col_values);
    println!("Unique Values: {:?}", unique_values);
    println!("Less than or equal to {}: {:?}", threshold, lt_eq);
}
