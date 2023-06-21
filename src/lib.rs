#[allow(dead_code)]

pub mod utils {
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
}

pub mod data {
    use rand::prelude::*;
    use std::collections::HashSet;

    /// A struct for storing data in a matrix format.
    #[derive(Debug)]
    pub struct Data {
        pub rows: usize,
        pub cols: usize,
        pub data: Vec<Vec<f64>>,
    }

    impl Data {
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
        /// let data: treerustler::data::Data = treerustler::data::Data::random_data(10, 3);
        /// assert_eq!(data.rows, 10);
        /// assert_eq!(data.cols, 3);
        /// ```
        pub fn from_random(rows: usize, cols: usize) -> Data {
            let mut rng: ThreadRng = rand::thread_rng();
            let data: Vec<Vec<f64>> = (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen_range(0.0..1.0)).collect())
                .collect();
            Data { rows, cols, data }
        }

        /// Generates a matrix from a string where rows are separeted by a semicolon and columns
        /// are separated by whitespace.
        ///
        /// # Arguments
        ///
        /// * `input` - A string containing the data
        ///
        /// # Examples
        ///
        /// ```
        /// let input: &str = "1 2 3; 4 5 6; 7 8 9";
        /// let data: treerustler::data::Data = treerustler::data::Data::from_string(input);
        /// assert_eq!(data.rows, 3);
        /// assert_eq!(data.cols, 3);
        /// assert_eq!(data.data[0][0], 1.0);
        /// ```
        pub fn from_string(input: &str) -> Data {
            let mut data: Vec<Vec<f64>> = Vec::new();
            let rows: Vec<&str> = input.split(";").collect();
            for row in rows {
                let entries: Vec<&str> = row.split_whitespace().collect();
                let mut row_entries: Vec<f64> = Vec::new();
                for entry in entries {
                    row_entries.push(entry.parse::<f64>().unwrap());
                }
                data.push(row_entries);
            }
            let row_size: usize = data.len();
            let col_size: usize = data[0].len();
            Data {
                rows: row_size,
                cols: col_size,
                data,
            }
        }

        /// Function to index column in Data Struct (dumb function to avoid impleting with std::ops::Index)
        ///
        /// # Arguments
        ///
        /// * `col_idx` - The index of the column to be returned
        ///
        /// # Examples
        ///
        /// ```
        /// let data: treerustler::data::Data = treerustler::data::Data::from_string("1 2 3; 4 5 6; 7 8 9");
        /// let col: Vec<f64> = data.get_col(0);
        /// assert_eq!(col, vec![1.0, 4.0, 7.0]);
        ///
        pub fn get_col(&self, col_idx: usize) -> Vec<f64> {
            // Maybe it's worth making Data col-major instead of row-major
            self.data
                .iter()
                .map(|row| *row.get(col_idx).unwrap())
                .collect()
        }

        /// Gets the sorted unique values in a column of the data matrix.
        ///
        /// # Arguments
        ///
        /// * `col_idx` - The index of the column to be returned
        ///
        /// # Examples
        ///
        /// ```
        /// let data: treerustler::data::Data = treerustler::data::Data::from_string("1 2 3; 4 5 6; 4 8 9");
        /// let unique_values: Vec<f64> = data.unique(0);
        /// println!("{:?}", unique_values);
        /// ```
        pub fn unique(&self, col: usize) -> Vec<f64> {
            let col_values: Vec<f64> = self.get_col(col);
            // Had to use this HashSet u64 trick because Hash or Eq Trait weren't implemented for f64
            let _temp: HashSet<u64> = col_values.iter().map(|&x| x.to_bits()).collect();
            // HashSet doesn't keep order so we first order the Vec<u64> and then convert back to Vec<f64>
            // sort_floats() is in nightly so we have to do this
            let mut _temp: Vec<u64> = _temp.into_iter().map(|x| x).collect();
            _temp.sort();
            _temp.into_iter().map(|x| f64::from_bits(x)).collect()
        }

        pub fn gt_eq(&self, col_idx: usize, threshold: f64) -> Vec<bool> {
            let col_values: Vec<f64> = self.get_col(col_idx);
            col_values.iter().map(|&x| x <= threshold).collect()
        }
    }
}
