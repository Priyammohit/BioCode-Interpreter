# DNA Sequence Classification Project

## Overview

This project focuses on the classification of DNA sequences from human, chimpanzee, and dog datasets using advanced machine learning techniques. The primary goal was to accurately categorize the DNA sequences into seven distinct classes, utilizing Python and popular data science libraries. The implemented model achieved outstanding performance, demonstrating a high level of accuracy, precision, recall, and F1 score.

###Deployment

The project has been deployed at https://biocode-interpreter.streamlit.app/.

## Table of Contents

- [Project Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Data Description](#data-description)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Key Features

- **Multi-Class Classification**: Efficiently classified DNA sequences into seven categories using optimized machine learning algorithms.
- **High Model Performance**: Achieved an accuracy of 98.4%, with precision, recall, and F1 score all at 0.984.
- **Confusion Matrix Analysis**: Detailed analysis of the classification results using a confusion matrix to evaluate model performance across all classes.

## Tech Stack

- **Programming Language**: Python
- **Libraries**: 
  - Scikit-learn: For implementing machine learning algorithms and performance evaluation.
  - Pandas: For data manipulation and analysis.
  - NumPy: For numerical computations.
  - Matplotlib & Seaborn: For data visualization.
  
## Data Description

The dataset used for this project contains DNA sequences from three species: humans, chimpanzees, and dogs. The sequences are labeled into seven different classes based on their characteristics. Each sample is represented by a series of features extracted from the DNA sequences.

### Classes:

1. **Class 0**: Human DNA
2. **Class 1**: Chimpanzee DNA
3. **Class 2**: Dog DNA
4. **Class 3**: [Provide specific descriptions for each class if available]
5. **Class 4**: [Provide specific descriptions for each class if available]
6. **Class 5**: [Provide specific descriptions for each class if available]
7. **Class 6**: [Provide specific descriptions for each class if available]

## Model Performance

The model was evaluated using several metrics, and the results were highly satisfactory:

- **Accuracy**: 98.4%
- **Precision**: 0.984
- **Recall**: 0.984
- **F1 Score**: 0.984

### Confusion Matrix:

| Predicted | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|-----------|---|---|---|---|---|---|---|
| Actual    |   |   |   |   |   |   |   |
| 0         | 99| 0 | 0 | 0 | 1 | 0 | 2 |
| 1         | 0 |104| 0 | 0 | 0 | 0 | 2 |
| 2         | 0 | 0 |78 | 0 | 0 | 0 | 0 |
| 3         | 0 | 0 | 0 |124| 0 | 0 | 1 |
| 4         | 1 | 0 | 0 | 0 |143| 0 | 5 |
| 5         | 0 | 0 | 0 | 0 | 0 |51 | 0 |
| 6         | 1 | 0 | 0 | 1 | 0 | 0 |263 |

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/dna-sequence-classification.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd dna-sequence-classification
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your data:** Ensure your DNA sequence data is in the format expected by the model.
2. **Run the classification script:**

    ```bash
    python classify_dna.py
    ```

   This script will load the dataset, preprocess the data, train the model, and output the classification results.

## Results

The trained model demonstrated high accuracy and robust performance across all classes. The confusion matrix and classification metrics indicate a well-balanced model with minimal misclassifications.

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an issue or submit a pull request. Ensure your code follows the project's coding standards and includes relevant tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

