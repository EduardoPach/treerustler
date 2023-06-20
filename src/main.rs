use rand::prelude::*;
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
    let data: utils::Data = utils::Data::random_data(10, 3);
    println!("{:#?}", data);
}
