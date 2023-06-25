#![allow(dead_code)]
use crate::data::Data;
use std::collections::{HashMap, HashSet};

pub mod utils;

#[derive(Debug)]
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

#[derive(Debug)]
pub struct DecisionTreeClassifier {
    root: Option<Box<Node>>,
    max_depth: usize,
    min_samples_split: usize,
}

impl DecisionTreeClassifier {
    fn build_tree(&self, x: &Data, y: &Vec<u8>, depth: usize) -> Node {
        // Check if we should stop
        if self.should_stop(depth, &y) {
            // Get the value for the leaf node
            let value: HashMap<u8, f64> = utils::get_probabilities(&y);
            // Create the leaf node
            return Node {
                feature: None,
                threshold: None,
                left: None,
                right: None,
                value: Some(value),
            };
        } else {
            // Find the best split
            let (best_feature, best_threshold) = utils::find_best_split(x, y);
            // Get the left and right indices
            let left_idx: Vec<bool> = x.lt_eq(best_feature, best_threshold);
            // Get the left and right data
            let split: utils::Split = utils::get_split(x, y, left_idx);
            // Build the left and right subtrees
            let left_node: Node =
                self.build_tree(&split.left_data.0, &split.left_data.1, depth + 1);
            let right_node: Node =
                self.build_tree(&split.right_data.0, &split.right_data.1, depth + 1);
            // Create the internal node
            return Node {
                feature: Some(best_feature),
                threshold: Some(best_threshold),
                left: Some(Box::new(left_node)),
                right: Some(Box::new(right_node)),
                value: None,
            };
        }
    }

    fn should_stop(&self, depth: usize, y: &Vec<u8>) -> bool {
        // Get sample size
        let n_samples = y.len();
        // Check if there's a single class
        let n_classes: usize = y.iter().collect::<HashSet<_>>().len();

        depth <= self.max_depth || n_samples < self.min_samples_split || n_classes == 1
    }

    fn traverse_tree(&self, x: Data) -> HashMap<u8, f64> {
        // Will Implement later
    }

    pub fn fit(x: Data, y: Vec<u8>) -> () {
        // Will Implement later
    }

    pub fn predict() -> Vec<u8> {
        // Will Implement later
    }
}
