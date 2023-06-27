#![allow(unused_imports)]
#![allow(dead_code)]
use rand::prelude::*;
use treerustler::{data, tree, tree::utils};

#[allow(dead_code)]
fn get_fake_data() -> (data::Data, Vec<u8>) {
    let x: data::Data = data::Data::from_string(&"1 3; 2 3; 3 1; 3 1; 2 3");
    let y: Vec<u8> = vec![0, 0, 1, 1, 2];
    (x, y)
}

fn main() {
    // let v: Vec<u8> = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
    // println!("{:?}", v);
    // let probabilities = utils::get_probabilities(&v);
    // println!("{:?}", probabilities);
    // let entropy: f64 = utils::entropy_loss(&v);
    // println!("Entropy Loss: {}", entropy);
    // let gini: f64 = utils::gini_index(&v);
    // println!("Gini Index: {}", gini);
    // let data: data::Data = data::Data::from_random(10, 3);
    // println!("{:#?}", data);

    // let d: data::Data =
    //     data::Data::from_string(&"1.1 2.4; 1.1 0.0; 5.5 3.1; 4.0 4.0; 4.0 1.25; 6.1 3.1");
    // let col_idx: usize = 1;
    // let threshold: f64 = 2.0;
    // let col_values: Vec<f64> = d.get_col(col_idx);
    // let unique_values: Vec<f64> = d.unique(col_idx);
    // let lt_eq: Vec<bool> = d.lt_eq(col_idx, threshold);

    // println!("Data = {:#?}", d);
    // println!("Data[:, {}] = {:?}", col_idx, col_values);
    // println!("Unique Values: {:?}", unique_values);
    // println!("Less than or equal to {}: {:?}", threshold, lt_eq);

    let (x, y) = get_fake_data();
    println!("X = {:#?}", x);
    println!("y = {:?}", y);

    let mut model = tree::DecisionTreeClassifier::new(1, 2);
    model.fit(&x, &y);
    println!("Model: {:#?}", model);
    let prediction = model.predict_proba(&x);
    println!("Prediction: {:#?}", prediction);
}
