# LSVT Voice Rehabilitation Classification

## Project Overview

This project aims to develop a robust machine learning model to classify the acceptability of patients' voices post-rehabilitation. Accurate classification of voices into acceptable and unacceptable categories is critical for evaluating the success of rehabilitation procedures and guiding further treatments. Our solution focuses on building a high-performance classification system to correctly replicate the experts' binary assessment. We experimented with default and tuned architectures for each model type, optimizing performance through hyperparameter tuning using grid search. The results were then compared using various performance metrics, including accuracy, precision, recall, F1 score and confusion matrices.

## Theoretical Background

Lee Silverman Voice Treatment (LSVT) is a speech therapy program that helps people with Parkinson's disease (PD) and other conditions improve their ability to speak and be understood. LSVT focuses on:

- Loudness: Recalibrating a person's perception of their own voice volume so they speak at a more normal level. 
- High-effort vocalization: Encouraging patients to make high-effort vocalizations. 
- Intensive therapy: Providing an intensive therapy plan.
- Calibration: Teaching patients how much effort is needed to raise their voice. 
- Quantification: Evaluating a patient's performance to motivate them.

## Methodology

1. **Problem Framing**: We identified the key challenge as classifying patient voices into acceptable (1) and unacceptable (2) categories, where identifying unacceptable cases (true positives) is critical.

2. **Data Preprocessing**: The dataset was cleaned, scaled, and split into training and testing sets to ensure fair evaluation of the models.

3. **Model Selection**: Three classification models were selected based on their general performance in binary classification tasks:

- Support Vector Classifier (SVC) for its ability to handle non-linear separations.
- Logistic Regression (LogReg) as a baseline for comparison.
- Multi-layer Perceptron (MLP) for its potential to model complex patterns.

4. **Hyperparameter Tuning**: We conducted a comprehensive grid search for each model to fine-tune hyperparameters and optimize performance. The goal was to improve metrics such as recall and F1 score, especially for the unacceptable class.

6. **Performance Comparison**: After tuning the models, we compared their performance across key metrics, with special attention to identifying true positives in the unacceptable class. Confusion matrices were also used to visualize classification errors.

## Solutions and Data Insights

### Default vs Tuned Model Architectures

We analyzed the impact of hyperparameter tuning on model performance by plotting the comparison of default and tuned architectures for all three models. The following insights emerged:

- Tuning SVC resulted in moderate improvements in precision and recall.
- Logistic Regression remained relatively stable, with slight gains in F1 score after tuning.
- MLPClassifier showed significant improvements post-tuning, particularly in recall and F1 score.

Key takeaway: MLP benefited the most from tuning, making it a strong candidate for this problem.

### Final Model Testing Accuracy Comparison

We compared the final testing accuracies of the SVC, LogReg and MLP models. The plot reveals that:

- SVC achieved an accuracy of 82.64%, with balanced precision and recall.
- LogReg yielded an accuracy of 83.45%, performing consistently but without excelling in recall.
- MLP demonstrated the highest accuracy of 86.27%, making it the most effective model overall.

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
