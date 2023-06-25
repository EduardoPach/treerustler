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
pub fn get_probabilities(v: &Vec<u8>) -> HashMap<&u8, f64> {
    let mut count_map: HashMap<&u8, f64> = HashMap::new();
    let vec_size: f64 = v.len() as f64;
    for element in v.iter() {
        *count_map.entry(element).or_insert(0.0) += 1.0 / vec_size;
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
