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
    pub fn from_data(data: Vec<Vec<f64>>) -> Data {
        let rows: usize = data.len();
        let cols: usize = data[0].len();
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

    pub fn get_rows(&self, row_idx: Vec<bool>) -> Data {
        let data_rows: Vec<Vec<f64>> = self
            .data
            .iter()
            .zip(row_idx.iter())
            .filter_map(|(row, &b)| if b { Some(row.clone()) } else { None })
            .collect();

        Data {
            rows: data_rows.len(),
            cols: self.cols,
            data: data_rows,
        }
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

    /// Returns a vector of booleans indicating whether the value in the column is less than the
    /// threshold.
    ///
    /// # Arguments
    ///
    /// * `col_idx` - The index of the column to be returned
    /// * `threshold` - The threshold to compare the column values to
    ///
    /// # Examples
    ///
    /// ```
    /// let data: treerustler::data::Data = treerustler::data::Data::from_string("1 2 3; 4 5 6; 4 8 9");
    /// let lt_eq: Vec<bool> = data.lt_eq(1, 4.0);
    /// assert_eq!(lt_eq, vec![true, false, false])
    /// ```
    pub fn lt_eq(&self, col_idx: usize, threshold: f64) -> Vec<bool> {
        let col_values: Vec<f64> = self.get_col(col_idx);
        col_values.iter().map(|&x| x <= threshold).collect()
    }
}
