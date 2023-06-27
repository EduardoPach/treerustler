# TreeRustler

TreeRustler is a simple implementation of a Decision Tree Classifier using the Rust programming language. This project serves as a beginner's exploration of Rust while building a decision tree classifier. The main modules in this project are the `data` module and the `tree` module.

## Data Module

The `data` module provides the `Data` struct, which represents the expected data type for the tree implementation. The `Data` struct holds the training features required for building the decision tree classifier. This module handles only the necessary data operations for the Decision Tree (i.e. no loading, preprocessing, or splitting yet!).

## Tree Module

The `tree` module contains the `DecisionTreeClassifier` struct. This struct represents the decision tree classifier and provides the following parameters:

- `max_depth`: Specifies the maximum depth of the decision tree. It controls how deep the tree can grow during training. Setting a smaller value can help prevent overfitting, but too small may result in underfitting.
- `min_samples_split`: Specifies the minimum number of samples required to split an internal node during training. It controls when to stop splitting nodes further. Setting a higher value can prevent overfitting, but too high may result in underfitting.

The `DecisionTreeClassifier` struct has the following methods:

- `fit(x: &Data, y: &Vec<u8>)`: Fits the decision tree classifier to the provided training data. This method trains the classifier using the feature from the `Data` struct and labels from a different vector.
- `predict_proba(x: &Data) -> Vec<f64>`: Predicts the class probabilities for the provided data using the trained decision tree. It returns a vector of probabilities for each class.

## Usage

To use the TreeRustler project, follow these steps:

1. Clone the repository: `git clone https://github.com/EduardoPach/treerustler.git`
2. Navigate to the project directory: `cd treerustler`
3. Make sure you have Rust installed. If not, install Rust from [https://www.rust-lang.org/](https://www.rust-lang.org/).
4. Load your data and convert your features data to the `Data` struct and your labels to a `Vec<u8>`.
5. Create an instance of `DecisionTreeClassifier` from the `tree` module, specifying the desired `max_depth` and `min_samples_split` values.
6. Call the `fit` method on the classifier instance, passing your training data.
7. Use the `predict_proba` method to predict class probabilities for new data points.

```rust
use treerustler::data::Data;
use treerustler::tree::DecisionTreeClassifier;

fn main() {
    // Load your features data and labels
    let x: data::Data = data::Data::from_string(&"1 3; 2 3; 3 1; 3 1; 2 3");
    let y: Vec<u8> = vec![0, 0, 1, 1, 2];

    // Create an instance of DecisionTreeClassifier
    let mut classifier = DecisionTreeClassifier::new(max_depth, min_samples_split);

    // Fit the classifier to the training data
    classifier.fit(&x, &y);

    // Predict class probabilities for new data points
    let probabilities = classifier.predict_proba(&x);

    // Process the predicted probabilities as needed
}
```
