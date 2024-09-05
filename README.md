# LSVT Voice Rehabilitation Classification

## Project Overview

This project aims to develop a robust machine learning model to classify the acceptability of patients' voices post-rehabilitation. Accurate classification of voices into acceptable and unacceptable categories is critical for evaluating the success of rehabilitation procedures and guiding further treatments. Our solution focuses on building a high-performance classification system in Python using three separate models to correctly replicate the experts' binary assessment. We experimented with default and tuned architectures for each model type, optimizing performance through hyperparameter tuning using grid search. The results were then compared using various performance metrics, including accuracy, precision, recall, F1 score and confusion matrices.

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

3. **Model Selection**: A diverse set of machine learning models was explored to classify voice rehabilitation success. Our selection was driven by a combination of model interpretability, performance, and their ability to handle different data structures and distributions.

4. **Hyperparameter Tuning**: We conducted a comprehensive grid search for each model to fine-tune hyperparameters and optimize performance. The goal was to improve metrics such as accuracy and F1 score.

6. **Performance Comparison**: After tuning the models, we selected the best architectures and compared their performance across key metrics, with special attention to identifying true positives in the unacceptable class. Confusion matrices were also used to visualize classification errors.

## Model Selection and Hyperparameter Tuning

### Initial Model Selection

We began with a broad range of models to compare the performance of their tuned and default architectures. Below is a summary of the models examined:

- **Dummy Classifiers**: We used several baseline classifiers to establish a minimal benchmark for performance. Specifically, we defined a) *Dummy1* and *Dummy2* for constant strategies, where predictions are always 1 or 2 respectively, b) *DummyMF* for always predicting the most frequent class, c) *DummyUni* for uniform random predictions and d) *DummyStr* for stratified predictions based on class distribution.
- **Gaussian Naive Bayes (GaussianNB)**: A probabilistic model using Gaussian distribution assumptions.
- **K-Nearest Neighbors (KNN)**: A distance-based algorithm that classifies based on the labels of nearby points in the dataset.
- **Support Vector Classifier (SVC)**: A powerful classifier that works by finding the optimal hyperplane separating the classes.
- **Logistic Regression (LogReg)**: A linear model used for binary classification.
- **Random Forest (RandForest)**: An ensemble learning method combining multiple decision trees to increase accuracy and reduce overfitting.
- **Decision Tree**: A model that splits the data into branches based on feature thresholds.
- **Gradient Boosting Classifier**: An ensemble model that builds decision trees sequentially to improve performance.
- **AdaBoost Classifier**: A boosting technique that adjusts the weights of observations and builds models iteratively.
- **Multi-Layer Perceptron (MLP)**: A type of artificial neural network known for its flexibility and ability to capture non-linear relationships.

### Hyperparameter Tuning

Each model's performance was improved by tuning key hyperparameters using a grid search approach. Below, we detail the key hyperparameters chosen for tuning and their role in model performance:

#### GaussianNB:

```
- var_smoothing: Controls the variance of the Gaussian smoothing applied to each feature. Smaller values result in more variance, helping avoid overfitting.
```

#### K-Nearest Neighbors (KNN):
```
- n_neighbors: Number of neighbors to consider for majority voting.
- weights: Determines whether the neighbors' votes are weighted by distance or uniformly.
- p: Power parameter for the Minkowski distance metric.
```

#### SVC:
```
C: Regularization parameter, controlling the trade-off between maximizing the margin and minimizing classification error.
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

#### MLP:
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
- F1 Score: The harmonic mean of precision and recall, providing a balanced measure, especially in imbalanced datasets.

The final models were selected based on their performance across these metrics. The three classification models that were selected are: a) *Support Vector Classifier (SVC)*, b) *Logistic Regression (LogReg)* and c) *Multi-layer Perceptron (MLP)*.

## Solutions and Data Insights

### Default vs Tuned Model Architectures

We analyzed the impact of hyperparameter tuning on model performance by plotting the comparison of default and tuned architectures for all three models. The following insights emerged:

- Tuning SVC resulted in moderate improvements in precision and recall.
- Logistic Regression remained relatively stable, with slight gains in F1 score after tuning.
- MLPClassifier showed significant improvements post-tuning, particularly in recall and F1 score.

![Comparison of Default and Tuned Architectures on Accuracy](./plots/tuned_vs_default_accuracy_plot.png)

![Comparison of Default and Tuned Architectures on F1-micro](./plots/tuned_vs_default_f1_micro_plot.png)

![Comparison of Default and Tuned Architectures on F1-macro](./plots/tuned_vs_default_f1_macro_plot.png)

Key takeaway: MLP benefited the most from tuning, making it a strong candidate for this problem.

### Final Model Testing Accuracy Comparison

We compared the final testing accuracies of the SVC, LogReg and MLP models. The plot reveals that:

- SVC achieved an accuracy of 82.64%, with balanced precision and recall.
- LogReg yielded an accuracy of 83.45%, performing consistently but without excelling in recall.
- MLP demonstrated the highest accuracy of 86.27%, making it the most effective model overall.

![Testing Accuracy of Selected Architectures](./plots/testing_accuracy_plot.png)

Key takeaway: MLPClassifier stands out with both high accuracy and recall, making it an excellent choice for the task.

### Confusion Matrix Analysis and F1 Scores
The confusion matrices for the final models reveal detailed classification performance for both acceptable (1) and unacceptable (2) categories:

- MLPClassifier shows a good F1 score of 87.18% and the highest recall at 89.47%. This means MLP is highly effective at identifying true positives for the unacceptable (2) class, which is crucial for our focus on voice rehabilitation cases that are deemed unsuccessful.

- SVC and LogReg had comparable F1 scores but slightly lower recall, making them less ideal for this specific problem, where unacceptable (2) cases need to be correctly identified more often.

Key takeaway: The MLPClassifier is the best option for identifying unacceptable voice rehabilitation outcomes, which may be the most critical aspect of this application. Its ability to maximize recall ensures fewer false negatives, leading to more accurate identification of problematic cases that require further intervention.

## Conclusion

This project demonstrated the power of machine learning in addressing complex, real-world challenges in voice rehabilitation classification. Through comprehensive model selection, tuning and evaluation, we found that MLPClassifier provided the most accurate and reliable performance, especially in identifying unacceptable cases, which could be crucial for improving patient outcomes. Further enhancements could involve adding more features or experimenting with ensemble techniques to boost model performance even further. The current solution, however, is robust and ready for deployment in clinical settings.

## Citation
Tsanas, A. (2014). LSVT Voice Rehabilitation [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52S4Z.
