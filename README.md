# 🧠 AlgoVision: Statistical Methods in AI 🚀

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive implementation of various machine learning algorithms from scratch, including Linear Regression, K-Nearest Neighbors, Decision Trees, and K-Means clustering.

## 📋 Project Overview

This repository contains implementations and analyses for Statistical Methods in AI (Spring 2025) course assignment. The project explores fundamental machine learning algorithms through practical applications on real-world datasets.

### ✨ Key Features

- 📊 Linear Regression from scratch with different Gradient Descent variants
- 🔍 K-Nearest Neighbors and Approximate Nearest Neighbor search implementations
- 🌲 Decision Tree algorithm implementation for fraud detection
- 🧩 SLIC (Simple Linear Iterative Clustering) implementation for image segmentation
- 📈 Comprehensive visualizations and performance analyses

## 🔧 Project Structure

```
├── 📓 AlgoVision.ipynb
├── 📄 Food_Delivery_Times.csv
├── 📄 text_embeddings.pth
├── 📄 test_embeddings.pth
├── 📄 train_embeddings.pth
├── 📄 test_labels.pth
├── 📄 train_labels.pth
├── 📄 input.mp4
├── 📄 segment.pth
├── 📄 frame_0000.jpg
├── 📁 images/
│   ├── 📄 out_20m_k200.jpg
        ...
├── 📁 frames/
│   ├── 📄 frame_0065.jpg
        ...
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 Report.pdf
```

## Detailed Implementation Analysis

### 1. Food Delivery Time Prediction

#### 1.1 Exploratory Data Analysis

Our EDA revealed several key patterns in the food delivery dataset:

- **Order Distribution:** Orders are uniformly distributed across all areas
- **Preparation Time Clusters:** Six distinct clusters in preparation time suggest different food categories
- **Delivery Time Distribution:** Average delivery time is around 56 minutes
- **Weather Impact:** Delivery times are highest in Snowy conditions, followed by Rainy, Foggy, and Windy, with Clear weather having the fastest deliveries
- **Traffic Impact:** High traffic significantly increases delivery time compared to Medium and Low traffic
- **Distance Correlation:** Strong positive correlation between distance and delivery time
- **Experience Factor:** Experienced couriers complete deliveries faster

#### 1.2 Linear Regression with Gradient Descent

We implemented three variants of gradient descent with the following parameters:
- Learning rate: 0.05
- Random and zero weight initialization
- MSE loss function
- Maximum iterations: 1000

**Gradient Descent Performance (Zero Initialization):**

| Method | Test MSE | R² Score | Train Loss | Validation Loss |
|--------|----------|---------|------------|----------------|
| Batch GD | 137.43 | 0.765 | 150.84 | 158.43 |
| Stochastic GD | 137.54 | 0.765 | 152.94 | 159.32 |
| Mini-Batch GD | 137.78 | 0.764 | 151.23 | 158.87 |

**Gradient Descent Performance (Random Initialization):**

| Method | Test MSE | R² Score | Train Loss | Validation Loss |
|--------|----------|---------|------------|----------------|
| Batch GD | 137.43 | 0.765 | 150.84 | 158.43 |
| Stochastic GD | 138.82 | 0.763 | 159.54 | 163.88 |
| Mini-Batch GD | 137.91 | 0.764 | 152.36 | 159.12 |

#### 1.3 Regularization Analysis

**Performance Metrics at λ = 0.5:**

| Method | Test MSE | R² Score |
|--------|----------|---------|
| Ridge Regression | 307.11 | 0.54 |
| Lasso Regression | 155.37 | 0.76 |

**Regularization Impact (λ = 0 to 1):**

| λ Value | Ridge Test MSE | Ridge R² | Lasso Test MSE | Lasso R² |
|---------|---------------|----------|---------------|----------|
| 0.00 | 137.43 | 0.765 | 137.43 | 0.765 |
| 0.22 | 199.76 | 0.693 | 137.48 | 0.765 |
| 0.33 | 238.53 | 0.634 | 139.54 | 0.762 |
| 0.50 | 307.11 | 0.540 | 155.37 | 0.760 |
| 0.75 | 370.54 | 0.442 | 165.42 | 0.756 |
| 1.00 | 387.76 | 0.397 | 169.94 | 0.753 |

### 2. Gradient Descent Algorithms: Comparison and Analysis

#### 2.1 Types of Gradient Descent Algorithms

**Batch Gradient Descent:**
- Uses the entire dataset to compute gradients before updating parameters
- Provides stable convergence but is computationally expensive for large datasets
- Best suited for convex, smooth functions

**Stochastic Gradient Descent (SGD):**
- Updates parameters using a single randomly chosen sample per iteration
- Faster for large datasets but exhibits high variance in updates
- Can escape local minima due to randomness

**Mini-Batch Gradient Descent:**
- Uses small random subsets of data for gradient computation
- Balances speed and stability between Batch GD and SGD
- Requires careful batch size selection

#### 2.2 Feature Influence on Delivery Time

Analysis of feature weights reveals:

- **High Impact Features:** Preparation time, Distance (km), and Weather conditions
- **Low Impact Features:** Time of day, Vehicle type, and Traffic level
- **Lasso Regularization Effect:** Effectively identified irrelevant features by setting their weights to near-zero
- **Ridge Regularization Effect:** Reduced influence of all features while retaining them in the model

**Feature Weight Comparison:**

| Feature | No Regularization | Ridge (λ=0.5) | Lasso (λ=0.5) |
|---------|-------------------|--------------|--------------|
| Preparation time | 8.476 | 6.832 | 7.942 |
| Distance (km) | 5.321 | 4.253 | 5.104 |
| Weather | 2.154 | 1.932 | 1.983 |
| Order Area | 1.876 | 1.521 | 1.754 |
| Courier Experience | -1.432 | -1.244 | -1.365 |
| Traffic Level | 0.876 | 0.654 | 0.087 |
| Time of day | 0.342 | 0.231 | 0.022 |
| Vehicle Type | 0.211 | 0.187 | 0.011 |

### 3. KNN and ANN Analysis

#### 3.1 KNN Implementation and Statistics

We implemented K-Nearest Neighbors for both classification and retrieval tasks:

**Text Embedding Classification Accuracy:**

| k Value | Accuracy |
|---------|----------|
| 1 | 0.9576 |
| 3 | 0.9612 |
| 5 | 0.9634 |
| 7 | 0.9589 |
| 9 | 0.9542 |

**Retrieval Performance Metrics:**

| Retrieval Type | Mean Reciprocal Rank | Precision@100 | Hit Rate |
|----------------|---------------------|--------------|----------|
| Text to Image | 1.0000 | 0.9740 | 1.0000 |
| Image to Image | 0.9348 | 0.8411 | 0.9996 |

#### 3.2 Approximate Nearest Neighbors: LSH Implementation

Locally Sensitive Hashing (LSH) was implemented with varying bit counts to analyze the trade-off between speed and accuracy:

**Impact of Bit Count on Performance:**

| Bit Count | MRR | Precision@100 | Hit Rate | Query Speed (it/s) |
|-----------|-----|--------------|----------|-------------------|
| 4 | 0.9210 | 0.7982 | 0.9991 | 47.48 |
| 8 | 0.9145 | 0.7478 | 0.9976 | 128.94 |
| 12 | 0.9064 | 0.6549 | 0.9919 | 452.17 |
| 16 | 0.8694 | 0.4298 | 0.9662 | 835.25 |

**Key Insights:**
- More bits result in faster queries but reduced accuracy
- 8 bits provides a good balance between performance and accuracy
- Higher bit counts create more specific buckets, potentially separating similar items

#### 3.3 IVF Implementation

Inverted File Index (IVF) was implemented as another ANN approach:

**IVF Performance Metrics:**

| nprobe Value | MRR | Precision@100 | Hit Rate | Avg. Comparisons |
|--------------|-----|--------------|----------|-----------------|
| 1 | 0.7654 | 0.3542 | 0.9321 | 1,200 |
| 5 | 0.8876 | 0.6423 | 0.9842 | 5,800 |
| 10 | 0.9087 | 0.7532 | 0.9923 | 11,400 |
| 20 | 0.9312 | 0.8276 | 0.9987 | 22,600 |

**Comparison of Search Techniques:**

| Method | MRR | Precision@100 | Hit Rate | Query Time |
|--------|-----|--------------|----------|------------|
| Brute Force | 0.9348 | 0.8411 | 0.9996 | Slow |
| LSH (8 bits) | 0.9145 | 0.7478 | 0.9976 | Medium |
| IVF (nprobe=10) | 0.9087 | 0.7532 | 0.9923 | Fast |

### 4. Crypto Fraud Detection

#### 4.1 Data Analysis and Preprocessing

The crypto transaction dataset was analyzed to identify patterns related to fraudulent activities:

**Key Correlations with FLAG (Fraud Indicator):**
- Time difference between first and last transaction (-0.25)
- Average minimum time between received transactions (-0.11)
- Total transactions (-0.13)

**Feature Distribution Observations:**
- Flagged accounts (FLAG=1) exhibit broader distributions across most features
- Unflagged accounts (FLAG=0) have more concentrated distributions
- Features like time difference and maximum received value show distinct differences between flagged and unflagged accounts

#### 4.2 Decision Tree Implementation

A decision tree classifier was implemented from scratch to detect fraudulent crypto transactions:

**Accuracy Comparison:**

| Model | Training Accuracy | Testing Accuracy | Validation Accuracy |
|-------|-------------------|-----------------|---------------------|
| Custom DT (max_depth=5) | 0.923 | 0.898 | 0.901 |
| Custom DT (max_depth=10) | 0.940 | 0.902 | 0.910 |
| Scikit-learn DT (max_depth=5) | 0.917 | 0.895 | 0.883 |
| Scikit-learn DT (max_depth=10) | 0.936 | 0.902 | 0.887 |

**Computation Time Comparison:**

| Model | Computation Time (seconds) |
|-------|----------------------------|
| Custom DT | 30-70 |
| Scikit-learn DT | 0.044 |

**Performance by Criterion:**

| Criterion | Accuracy (Custom) | Accuracy (Scikit-learn) |
|-----------|-------------------|------------------------|
| Entropy | 0.902 | 0.898 |
| Gini | 0.901 | 0.897 |

**Feature Importance (Top 5):**

| Feature | Importance |
|---------|------------|
| Time Diff between first and last (Mins) | 0.312 |
| total ether received | 0.187 |
| total ether balance | 0.154 |
| Unique Received From Addresses | 0.098 |
| max value received | 0.076 |

### 5. K-Means and SLIC Implementation

#### 5.1 SLIC Algorithm Results

The Simple Linear Iterative Clustering (SLIC) algorithm was implemented for image segmentation:

**Parameter Impact Analysis:**

| Parameter | Effect |
|-----------|--------|
| Number of clusters (k) | Higher k values create smaller superpixels with increased computational cost |
| Compactness (m) | Higher m produces more regular shapes; lower m adapts better to edges |
| Color space | RGB preserves colors as they appear; LAB is perceptually uniform |

**SLIC Performance with Different Parameters:**

| m Value | k Value | Segmentation Quality | Computational Cost |
|---------|---------|----------------------|-------------------|
| 10 | 200 | Good edge detection | Moderate |
| 20 | 200 | More regular shapes | Moderate |
| 20 | 3000 | Detailed, small clusters | High |

#### 5.2 Video Segmentation Optimization

The SLIC algorithm was extended for video segmentation with temporal optimization:

**Key Optimizations:**
1. Using previous frame's cluster centers as starting points
2. Processing frames in local regions instead of the entire image
3. Implementing vectorized operations for faster distance calculations
4. Employing early stopping when changes between iterations are small
5. Restricting search space to local neighborhoods

**Optimization Results:**

| Metric | Before Optimization | After Optimization | Improvement |
|--------|---------------------|-------------------|-------------|
| Iterations per frame | 10 | 3.64 | 63.6% |
| Processing time per frame | High | Moderate | Significant |
| Segmentation quality | Baseline | Maintained | No loss |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required packages listed in `requirements.txt`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/AlgoVision.git
   cd AlgoVision
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open and run the notebooks in the `notebooks/` directory.

## 📝 Usage Examples

### Linear Regression with Gradient Descent

```python
from src.linear_regression import LinearRegression

# Initialize model
model = LinearRegression(learning_rate=0.05, num_iterations=1000, 
                         gd_type='mini-batch', batch_size=32)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mse, r2 = model.evaluate(X_test, y_test)
```

### K-Nearest Neighbors

```python
from src.knn import KNN

# Initialize KNN model
knn = KNN(k=5, distance_metric='cosine')

# Train model
knn.fit(X_train, y_train)

# Make predictions
predictions = knn.predict(X_test)

# Calculate accuracy
accuracy = knn.accuracy(predictions, y_test)
```

### Decision Tree

```python
from src.decision_tree import DecisionTree

# Initialize decision tree
dt = DecisionTree(max_depth=5, min_samples_split=2, criterion='entropy')

# Train model
dt.fit(X_train, y_train)

# Make predictions
predictions = dt.predict(X_test)

# Calculate feature importance
importance = dt.feature_importance()
```

### SLIC Image Segmentation

```python
from src.slic import SLIC
import cv2

# Load image
image = cv2.imread('data/images/sample.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize SLIC
slic = SLIC(num_clusters=200, compactness=10, max_iterations=10)

# Perform segmentation
segments = slic.segment(image)

# Visualize results
segmented_image = slic.visualize_segments(image, segments)
```

## 📊 Conclusion

This project demonstrates the implementation and evaluation of fundamental machine learning algorithms from scratch. The implementations provide insights into the inner workings of these algorithms and their performance characteristics on real-world datasets.

Key learnings:
- Mini-batch gradient descent offers the best balance between convergence speed and stability
- Lasso regularization provides better feature selection for the food delivery dataset
- LSH significantly improves retrieval speed with minimal accuracy loss
- Decision trees achieve high accuracy for fraud detection with proper hyperparameter tuning
- SLIC provides effective image segmentation with significant optimization potential through temporal information

## 👥 Contributors

- Mayank Mittal (2022101094)


## 🙏 Acknowledgements

- Statistical Methods in AI course instructors and TAs
- [OpenAI](https://openai.com/) for CLIP embeddings
- [Scikit-learn](https://scikit-learn.org/) for reference implementations
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization tools
