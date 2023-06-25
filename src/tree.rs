#![allow(dead_code)]

use crate::data::Data;
use std::collections::HashMap;

pub mod utils;

struct Node {
    feature: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    value: Option<HashMap<u8, f64>>,
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.value.is_some()
    }
}

struct Split {
    left_data: (Data, Vec<u8>),
    right_data: (Data, Vec<u8>),
}

pub struct DecisionTreeClassifier {
    root: Option<Box<Node>>,
    max_depth: usize,
    min_samples_split: usize,
}

// fn find_best_split(x: Data, y: Vec<u8>) -> (usize, f64) {
//     // Will Implement later
// }

// fn get_split(x: Data, y: Vec<u8>, left_idx: Vec<bool>) -> Split {
//     // Will Implement later
// }

// impl DecisionTreeClassifier {
//     fn build_tree(&self, x: Data, y: Vec<u8>, depth: usize) -> Node {
//         // Get info about data to check if we should stop
//         let n_samples: usize = y.len();
//         // Check if we should stop
//         if self.should_stop(depth, n_samples) {
//             // Get the value for the leaf node
//             let value: HashMap<u8, f64> = utils::get_probabilities(&y);
//             // Create the leaf node
//             return Node {
//                 feature: None,
//                 threshold: None,
//                 left: None,
//                 right: None,
//                 value: Some(value),
//             };
//         } else {
//             // Find the best split
//             let (best_feature, best_threshold) = self.find_best_split(&x, &y);
//             // Get the left and right indices
//             let left_idx: Vec<bool> = x.lt_eq(best_feature, best_threshold);
//             // Get the left and right data
//             let split: Split = get_split(x, y, left_idx);
//         }
//     }

//     fn should_stop(&self, depth: usize, n_samples: usize) -> bool {
//         depth <= self.max_depth || n_samples < self.min_samples_split
//     }

//     fn traverse_tree(&self, x: Data) -> HashMap<u8, f64> {
//         // Will Implement later
//     }

//     pub fn fit(x: Data, y: Vec<u8>) -> () {
//         // Will Implement later
//     }

//     pub fn predict() -> Vec<u8> {
//         // Will Implement later
//     }
// }
