#[allow(dead_code)]

pub mod utils {
    use rand::prelude::*;

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

    #[derive(Debug)]
    pub struct Data {
        pub rows: usize,
        pub cols: usize,
        pub data: Vec<Vec<f64>>,
    }

    impl Data {
        pub fn new(rows: usize, cols: usize, data: Vec<Vec<f64>>) -> Data {
            Data { rows, cols, data }
        }

        /// Generates a random matrix of size (rows, cols) with values in the range [0, 1).
        ///
        /// # Arguments
        ///
        /// * `rows` - The number of rows in the matrix
        /// * `cols` - The number of columns in the matrix
        ///
        /// # Examples
        ///
        /// ```
        /// let data: treerustler::utils::Data = treerustler::utils::Data::random_data(10, 3);
        /// assert_eq!(data.rows, 10);
        /// assert_eq!(data.cols, 3);
        /// ```
        pub fn random_data(rows: usize, cols: usize) -> Data {
            let mut rng: ThreadRng = rand::thread_rng();
            let data: Vec<Vec<f64>> = (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(0.0..1.0)).collect())
                .collect();
            Data::new(rows, cols, data)
        }
    }
}
