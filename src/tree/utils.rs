use crate::data::Data;
use std::collections::HashMap;

/// Computes the relative frequency of each unique element in a vector
///  categorical vector.
///
/// # Arguments
///
/// * `v` - A vector of u8 values
///
/// # Examples
///
/// ```
/// let v: Vec<u8> = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
/// let probabilities = treerustler::utils::get_probabilities(&v);
/// assert_eq!(probabilities.get(&0).unwrap(), &0.5);
/// assert_eq!(probabilities.get(&1).unwrap(), &0.5);
/// ```
pub fn get_probabilities(v: &Vec<u8>) -> HashMap<u8, f64> {
    let mut count_map: HashMap<u8, f64> = HashMap::new();
    let vec_size: f64 = v.len() as f64;
    for element in v.iter() {
        *count_map.entry(*element).or_insert(0.0) += 1.0 / vec_size;
    }
    count_map
}

/// Computes the entropy loss of a categorical vector.
///
/// # Arguments
///
/// * `v` - A vector of u8 values
///
/// # Examples
///
/// ```
/// let v: Vec<u8> = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
/// let entropy: f64 = treerustler::utils::entropy_loss(&v);
/// assert_eq!(entropy, 1.0);
/// ```
pub fn entropy_loss(v: &Vec<u8>) -> f64 {
    let probabilities = get_probabilities(v);
    probabilities.values().map(|&p| -p * p.log2()).sum()
}

/// Computes the gini index of a categorical vector.
///
/// # Arguments
///
/// * `v` - A vector of u8 values
///
/// # Examples
///
/// ```
/// let v: Vec<u8> = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
/// let gini: f64 = treerustler::utils::gini_index(&v);
/// assert_eq!(gini, 0.5);
/// ```
pub fn gini_index(v: &Vec<u8>) -> f64 {
    let probabilities = get_probabilities(v);
    let p_2: f64 = probabilities.values().map(|&p| p.powi(2)).sum();
    1.0 - p_2
}

pub struct Split {
    pub left_data: (Data, Vec<u8>),
    pub right_data: (Data, Vec<u8>),
}

fn split_y(y: &Vec<u8>, idx: &Vec<bool>) -> (Vec<u8>, Vec<u8>) {
    let left_y: Vec<u8> = y
        .iter()
        .zip(idx.iter())
        .filter_map(|(&y, &b)| if b { Some(y) } else { None })
        .collect();
    let right_y: Vec<u8> = y
        .iter()
        .zip(idx.iter())
        .filter_map(|(&y, &b)| if !b { Some(y) } else { None })
        .collect();
    (left_y, right_y)
}

pub fn find_best_split(x: &Data, y: &Vec<u8>) -> (usize, f64) {
    let n_cols: usize = x.cols;
    let mut col_idx: usize = 0;

    let mut best_feature: usize = 0;
    let mut best_threshold: f64 = 0.0;
    let mut best_score: f64 = std::f64::INFINITY;
    let len: f64 = y.len() as f64;

    while col_idx < n_cols {
        let unique_values: Vec<f64> = x.unique(col_idx);
        for threshold in unique_values {
            let lt_eq: Vec<bool> = x.lt_eq(col_idx, threshold);
            let (l_y, r_y) = split_y(&y, &lt_eq);

            let l_weight: f64 = l_y.len() as f64 / len;
            let r_weight: f64 = r_y.len() as f64 / len;

            let left_score: f64 = gini_index(&l_y);
            let right_score: f64 = gini_index(&r_y);
            let score: f64 = l_weight * left_score + r_weight * right_score;
            if score < best_score {
                best_feature = col_idx;
                best_threshold = threshold;
                best_score = score;
            }
        }

        col_idx += 1;
    }

    (best_feature, best_threshold)
}

pub fn get_split(x: &Data, y: &Vec<u8>, left_idx: Vec<bool>) -> Split {
    let not_left_idx: Vec<bool> = left_idx.iter().map(|&v| !v).collect();
    let (l_y, r_y) = split_y(y, &left_idx);
    let l_data: Data = x.get_rows(left_idx);
    let r_data: Data = x.get_rows(not_left_idx);
    Split {
        left_data: (l_data, l_y),
        right_data: (r_data, r_y),
    }
}
