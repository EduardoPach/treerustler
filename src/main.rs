#[allow(dead_code)]
use rand::prelude::*;
use std::collections::HashSet;
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

    // let mut f: Vec<f64> = vec![1.0, 1.0, 2.0, 2.5, 3.3, 3.3, 4.0];
    // let temp_f: Vec<u64> = f.iter().map(|&value| value.to_bits()).collect();
    // let unique_values: HashSet<u64> = temp_f.into_iter().collect();
    // let un: Vec<f64> = unique_values
    //     .into_iter()
    //     .map(|value: u64| f64::from_bits(value))
    //     .collect();
    // println!("{:?}", un);

    let d: data::Data =
        data::Data::from_string(&"1.1 2.4; 1.1 0.0; 5.5 3.1; 4.0 4.0; 4.0 1.25; 6.1 3.1");
    let col_idx: usize = 1;
    let col_values: Vec<f64> = d.get_col(col_idx);
    let unique_values: Vec<f64> = d.unique(col_idx);

    println!("Data = {:#?}", d);
    println!("Data[:, {}] = {:?}", col_idx, col_values);
    println!("Unique Values: {:?}", unique_values);
}
