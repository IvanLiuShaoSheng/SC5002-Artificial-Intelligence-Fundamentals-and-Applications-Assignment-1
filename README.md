# SC5002 - Lab Assignment Part 2: Low-Code ML Template

This document provides a detailed breakdown of the steps performed in the Google Colab script for the SC5002 Lab Assignment Part 2. The script demonstrates how to load a dataset, preprocess it, train machine learning models, evaluate them using cross-validation, and visualize the results.

---

## Prerequisites

The following Python libraries are required to run the script:
- `numpy`: Numerical computing
- `pandas`: Data manipulation and analysis
- `matplotlib.pyplot`: Data visualization
- `scikit-learn`: Machine learning library (used for datasets, model selection, models, and preprocessing)

## Step-by-Step Breakdown

### 1. Dataset Setup (TASK 2.1)

The first step is to load and prepare the data.

1.  **Loading the Dataset:** The script uses `fetch_california_housing(as_frame=True)` from `sklearn.datasets` to load the California Housing dataset. Setting `as_frame=True` loads the data directly into a pandas DataFrame.
2.  **Sampling Data:** To ensure faster execution for the lab assignment, the script samples a subset of the dataset: `df = housing.frame.sample(n=1000, random_state=42)`. We take 1000 random records and set a `random_state` to ensure reproducibility.
3.  **Viewing Data:** It prints the first 5 rows of the sampled dataset using `df.head()` to allow for a quick inspection of the features and target variable.
4.  **Separating Features and Target:**
    *   **Features (`X`):** The features are obtained by dropping the target variable column: `X = df.drop(columns=['MedHouseVal'])`.
    *   **Target (`y`):** The target variable to be predicted is the median house value: `y = df['MedHouseVal']`.
5.  **Data Preprocessing (Scaling):** Because features might have different scales (e.g., income vs. number of rooms), they are standardized using `StandardScaler()`. This centers the data around a mean of 0 with a standard deviation of 1 (`X_scaled = scaler.fit_transform(X)`).
6.  **Train-Test Split:** The scaled dataset is split into training and testing sets using `train_test_split`.
    *   `test_size=0.2`: 20% of the data is set aside for testing, and 80% is used for training.
    *   `random_state=42`: Ensures the split is consistent across runs.

### 2. Model Training & Cross-Validation (TASK 2.2)

In this task, we train two types of regression models and evaluate their performance.

1.  **Linear Regression:**
    *   A standard `LinearRegression()` model is instantiated.
    *   **Cross-Validation:** To get a robust estimate of the model's performance on unseen data, 5-fold cross-validation is used (`cross_val_score(lin_reg, X_train, y_train, cv=5, scoring='r2')`). The training data is split into 5 parts, the model is trained on 4 parts and evaluated on the 1 remaining part, repeating this process 5 times.
    *   The evaluation metric used is the **$R^2$ (R-squared) score**, which measures the proportion of variance in the target variable explained by the model. The mean score across the 5 folds is calculated and printed.
2.  **Ridge Regression Experiments:**
    *   Ridge Regression is a type of regularized linear model. The regularization strength is controlled by the hyperparameter `alpha`.
    *   An array of different alpha values is defined: `alphas = [0.1, 10.0, 1000.0]`.
    *   A loop iterates through these alpha values. For each value, a `Ridge(alpha=a)` model is created, and its performance is evaluated using the same 5-fold cross-validation setup.
    *   The mean $R^2$ scores for each alpha are printed and stored in a dictionary (`ridge_results`) for later comparison.

### 3. Visualization (TASK 2.3)

The final step is to visually compare the performance of the trained models.

1.  **Figure Setup:** A `matplotlib` figure is created with a specific size (`figsize=(9, 5)`).
2.  **Data Preparation:** The model names (for the x-axis) and their corresponding mean $R^2$ scores (for the y-axis) are collected into lists (`models` and `scores`).
3.  **Bar Chart Creation:** A bar chart is generated using `plt.bar()`. Different colors are assigned to make the bars visually distinct.
4.  **Formatting:**
    *   The y-axis is labeled, and a bold title is added to the chart.
    *   The y-axis limits are set from 0 to 1.0 (since $R^2$ is generally between 0 and 1, though it can be negative).
    *   A subtle y-axis grid is added for easier reading of the values.
5.  **Adding Value Labels:** A loop iterates through each bar to add the exact $R^2$ score as a text label just above the bar.
6.  **Display:** Finally, `plt.show()` is called to render the plot.

---

## Summary

This template provides a foundational workflow for evaluating machine learning models. It emphasizes the importance of data scaling, the robustness of cross-validation for evaluation, and the impact of hyperparameter tuning (alpha in Ridge Regression) on model performance, concluding with a clear visual comparison of the results.
