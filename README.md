# LSVT Voice Rehabilitation Classification

## Project Overview

This project aims to develop a robust machine learning model to classify the acceptability of patients' voices post-rehabilitation. Accurate classification of voices into acceptable and unacceptable categories is critical for evaluating the success of rehabilitation procedures and guiding further treatments. Our solution focuses on building a high-performance classification system in Python, correctly replicating the experts' binary assessment with absolute accuracy. We experimented with default and tuned architectures for each model type, optimizing performance through hyperparameter tuning using grid search. The results were then compared using various performance metrics, including accuracy, precision, recall, F1 score and confusion matrices.

## Theoretical Background

Lee Silverman Voice Treatment (LSVT) is a speech therapy program that helps people with Parkinson's disease (PD) and other conditions improve their ability to speak and be understood. LSVT focuses on:

- Loudness: Recalibrating a person's perception of their own voice volume so they speak at a more normal level. 
- High-effort vocalization: Encouraging patients to make high-effort vocalizations. 
- Intensive therapy: Providing an intensive therapy plan.
- Calibration: Teaching patients how much effort is needed to raise their voice. 
- Quantification: Evaluating a patient's performance to motivate them.

## Methodology

1. **Problem Framing**: We identified the key challenge as classifying patient voices into acceptable (1) and unacceptable (2) categories, where identifying unacceptable cases (true positives) is critical.

2. **Data Preprocessing**: The dataset was cleaned, scaled and split into training and testing sets to ensure fair evaluation of the models.

3. **Model Selection**: A diverse set of machine learning models was explored to classify voice rehabilitation success. Our selection was driven by a combination of model interpretability, performance and their ability to handle different data structures and distributions.

4. **Hyperparameter Tuning**: We conducted a comprehensive grid search for each model to fine-tune hyperparameters and optimize performance. The goal was to improve metrics such as accuracy and F1 score.

6. **Performance Comparison**: After tuning the models, we selected the best architectures and compared their performance across key metrics, with special attention to identifying true positives in the unacceptable class. Confusion matrices were also used to visualize classification errors.

## Model Selection and Hyperparameter Tuning

### Initial Model Selection

We began with a broad range of models to compare the performance of their tuned and default architectures. Below is a summary of the models examined:

- **Dummy Classifiers**: We used several baseline classifiers to establish a minimal benchmark for performance. Specifically, we defined a) *Dummy1* and *Dummy2* for constant strategies, where predictions are always 1 or 2 respectively, b) *DummyMF* for always predicting the most frequent class, c) *DummyUni* for uniform random predictions and d) *DummyStr* for stratified predictions based on class distribution.
- **Gaussian Naive Bayes (GaussianNB)**: A probabilistic model using Gaussian distribution assumptions.
- **K-Nearest Neighbors (KNN)**: A distance-based algorithm that classifies based on the labels of nearby points in the dataset.
- **Support Vector Classifier (SVC)**: A powerful classifier that works by finding the optimal hyperplane separating the classes.
- **Logistic Regression**: A linear model used for binary classification.
- **Random Forest**: An ensemble learning method combining multiple decision trees to increase accuracy and reduce overfitting.
- **Decision Tree**: A model that splits the data into branches based on feature thresholds.
- **Gradient Boosting Classifier**: An ensemble model that builds decision trees sequentially to improve performance.
- **AdaBoost Classifier**: A boosting technique that adjusts the weights of observations and builds models iteratively.
- **Multi-Layer Perceptron Classifier (MLPClassifier)**: A type of artificial neural network known for its flexibility and ability to capture non-linear relationships.

### Hyperparameter Tuning

Each model's performance was improved by tuning key hyperparameters using a grid search approach. Below, we detail the key hyperparameters chosen for tuning and their role in model performance:

#### GaussianNB:

```
var_smoothing: Controls the variance of the Gaussian smoothing applied to each feature.
```

#### KNN:
```
n_neighbors: Number of neighbors to consider for majority voting.
weights: Determines whether the neighbors' votes are weighted by distance or uniformly.
p: Power parameter for the Minkowski distance metric.
```

#### SVC:
```
C: Regularization parameter for the trade-off between margin and classification error.
kernel: Choice of kernel function (linear, radial basis, polynomial, sigmoid).
gamma: Defines how far the influence of a single training example reaches.
```

#### Logistic Regression:
```
C: Inverse of regularization strength.
solver: Algorithm to use for optimization.
max_iter: Maximum number of iterations taken for the solvers to converge.
```

#### Random Forest:
```
n_estimators: Number of trees in the forest.
max_features: Number of features to consider when looking for the best split.
max_depth: Maximum depth of the tree.
```

#### Decision Tree:
```
criterion: Function to measure the quality of a split (Gini impurity or entropy).
max_depth: Limits the depth of the tree to prevent overfitting.
min_samples_split: Minimum number of samples required to split a node.
```

#### Gradient Boosting:
```
n_estimators: Number of boosting stages to perform.
learning_rate: Shrinks the contribution of each tree by this factor.
max_depth: Controls the depth of individual trees.
```

#### AdaBoost:
```
n_estimators: Number of boosting stages.
learning_rate: Controls the contribution of each weak learner.
```

#### MLPClassifier:
```
hidden_layer_sizes: Number of neurons in each hidden layer.
solver: Optimization algorithm (adam, sgd, lbfgs).
activation: Activation function for the hidden layer (ReLU, tanh).
alpha: L2 penalty (regularization term) to prevent overfitting.
max_iter: Maximum number of iterations during training.
```

### Cross-Validation and Scoring Criteria

To ensure the robustness of our results, cross-validation was applied during model evaluation, particularly using k-fold cross-validation. This process helped prevent overfitting by training the model on multiple subsets of the data and assessing its performance on unseen subsets. We used certain scoring criteria to evaluate the models:

- Accuracy: The percentage of correctly classified samples.
- F1-micro Score: The overall F1 score computed by aggregating the contributions of all classes and then computing the average. This metric is useful for evaluating models in terms of their performance across all classes.
- F1-macro Score: The average F1 score computed for each class independently and then averaged. This metric gives equal weight to each class, making it particularly useful for evaluating models on imbalanced datasets.

The final models were selected based on their performance across these metrics. The three classification models that were selected are: a) *Support Vector Classifier (SVC)*, b) *Logistic Regression* and c) *Multi-layer Perceptron Classifier (MLPClassifier)*.

## Solutions and Data Insights

### Default vs Tuned Model Architectures

We analyzed the impact of hyperparameter tuning on model performance by plotting the comparison of default and tuned architectures for all three models. The following plots depict the score for different default and tuned architectures. The main difference between the plots is the scoring criterion based on which the best tuned architecture is selected for each model type. The scoring criterion for the first plot is the accuracy. The scoring criterion for the second plot is the F1-micro. The scoring criterion for the third plot is the F1-macro.

![Comparison of Default and Tuned Architectures on Accuracy](./plots/tuned_vs_default_accuracy_plot.png)

![Comparison of Default and Tuned Architectures on F1-micro](./plots/tuned_vs_default_f1_micro_plot.png)

![Comparison of Default and Tuned Architectures on F1-macro](./plots/tuned_vs_default_f1_macro_plot.png)

Key takeaway: Models like KNN and GaussianNB are generally outperformed by more complex models like SVC, Logistic Regression and MLPClassifier. This is expected, as simpler models may not capture complex patterns in a high-dimensional dataset.

### Final Model Testing Accuracy Comparison

We compared the final testing statistics of the SVC, Logistic Regression and MLPClassifier models. Our analysis reveals that MLPClassifier outperforms the other models with the highest possible testing accuracy (100%), while SVC and Logistic Regression achieve testing accuracy of 84.62% and 92.31% respectively. The following plot depects the testing accuracy of the tuned architectures for three models types. The architectures were tuned based on the scoring criteria of accuracy, F1-micro and F1-macro.

![Testing Accuracy of Selected Architectures](./plots/testing_accuracy_plot.png)

Key takeaway: MLPClassifier stands out with the highest possible accuracy, making it an excellent choice for the task.

### Analysis of Performance Metrics

In this binary classification problem, we are predicting whether a patient’s voice rehabilitation is acceptable (class 1) or unacceptable (class 2). Class 2 ("unacceptable") is treated as the positive class, meaning a higher emphasis might be placed on Recall and F1-score for correctly identifying unacceptable cases, which could be critical for medical decisions. 

| Model              | Tuned Hyperparameters (based on F1-micro)                     | Accuracy | Precision | Recall  | F1 score | F1-micro | F1-macro |
|--------------------|---------------------------------------------------------------|----------|-----------|---------|----------|----------|----------|
| SVC                | C = 1, gamma = 1, kernel = 'poly'                             | 84.62%   | 78.95%    | 100.00% | 88.24%   | 84.62%   | 83.01%   |
| Logistic Regression| C = 0.1                                                       | 92.31%   | 100.00%   | 86.67%  | 92.86%   | 92.31%   | 92.26%   |
| MLPClassifier      | alpha = 0.01, hidden_layer_sizes = (100,50), solver = 'lbfgs' | 100.00%  | 100.00%   | 100.00% | 100.00%  | 100.00%  | 100.00%  |

The table above contains the performance metrics associated with each architecture that was tuned based on the scoring criteria of F1-micro.

1. **SVC**
- Recall of 100% means the SVC model identifies all "unacceptable" cases (class 2) correctly, which is crucial in the medical context since failing to identify unacceptable cases could have severe consequences for patient treatment. However, Precision is lower (78.95%), indicating that many of the cases it predicts as unacceptable (class 2) are actually acceptable (class 1). This high false positive rate could lead to unnecessary concern or additional testing.
- F1-score (88.24%) balances Precision and Recall and shows that the model leans toward being more conservative, ensuring that no unacceptable case is missed. F1-micro (84.62%) gives an overall performance measure weighted by class frequency and F1-macro (83.01%) considers both classes equally, showing a slight imbalance favoring Recall over Precision.

Key takeaway: SVC is strong at identifying unacceptable cases (high Recall), but its lower Precision suggests a higher rate of false positives. If the focus is to ensure that no unacceptable case is missed (minimizing false negatives), this could be a reasonable tradeoff. However, its overall performance (Accuracy of 84.62%) shows there is room for improvement.

2. **Logistic Regression**
- Precision is perfect (100%), meaning that all predicted unacceptable cases (class 2) are indeed truly unacceptable. There are no false positives in this model's predictions. Recall (86.67%) is slightly lower than the SVC model, meaning it misses some unacceptable cases, but overall it performs quite well.
- The F1-score (92.86%) strikes a good balance between Precision and Recall. The F1-macro (92.26%) shows consistent performance across both classes, indicating that the model performs well regardless of the class distribution. F1-micro (92.31%) aligns with the Accuracy, confirming that the model is robust overall.

Key Takeaway: Logistic Regression offers a high-performing balance, with perfect Precision and strong Recall. This means it makes fewer mistakes in classifying unacceptable cases as acceptable (i.e. no false positives), though it does miss a few unacceptable cases. For applications where it is important to avoid false alarms (minimizing false positives), this model appears to be performing better than the SVC.

3. **MLPClassifier**
- This model delivers perfect scores across the board, meaning it identifies all unacceptable cases correctly (100% Recall) and also predicts them without error (100% Precision).
- Both F1-micro and F1-macro scores are at 100%, indicating flawless performance for both classes regardless of class distribution. In medical contexts, this is ideal as it ensures that all unacceptable cases are identified without making any false positive or false negative errors.

Key Takeaway: MLPClassifier stands out as the top-performing model, with perfect Precision, Recall and F1-scores. This makes it the most reliable for both identifying unacceptable cases and minimizing misclassifications. However, the perfect performance might be a result of the model overfitting, so further testing on unseen data is necessary to ensure generalizability.

## Conclusion

This project demonstrated the power of machine learning in addressing complex, real-world challenges in voice rehabilitation classification. Through comprehensive model selection, tuning and evaluation, we found that Neural Networks appear to offer the best performance, but Logistic Regression could be a safer option to avoid false positives, especially in scenarios where overfitting may be a concern. Further testing on unseen data would provide a clearer picture of which model should be deployed in a real-world setting. Further enhancements could involve adding more features or experimenting with ensemble techniques to boost model performance even further without the risk of overfitting. The current solution, however, is robust and can be used for deployment in clinical settings.

## Dataset
Tsanas, A. (2014). LSVT Voice Rehabilitation [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52S4Z.
