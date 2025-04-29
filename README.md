# Loan Approval Prediction Using Machine Learning

## Project Overview

This project investigates the application of machine learning algorithms to predict loan approval outcomes based on demographic and financial attributes of applicants. The study evaluates the effectiveness of Support Vector Machine (SVM), Decision Tree, and Random Forest classifiers in classifying loan applications as approved or rejected. The research aims to enhance objectivity, accuracy, and efficiency in financial risk assessment and decision-making.

## Research Objectives

- To construct predictive models that determine loan approval status using supervised machine learning algorithms.
- To perform appropriate data preprocessing, including encoding, normalization, and class balancing techniques.
- To analyze and visualize the distribution and correlation of variables through exploratory data analysis (EDA).
- To apply hyperparameter tuning to improve model performance and generalizability.
- To evaluate and compare models using standard metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
- To identify the most influential features affecting model decisions and potential real-world implications.

## Dataset Description

The dataset used in this project was obtained from Kaggle and contains anonymized records of 2,000 loan applicants. Features include Age, Income, Credit Score, Loan Amount, Loan Term, and Employment Status. The target variable, Loan_Approved, is a binary classification indicating whether the loan was approved (1) or rejected (0). The dataset complies with ethical standards and is publicly available for educational use under a Creative Commons Attribution license.

## Data Preprocessing

The preprocessing steps applied to the dataset include:

- Cleaning and standardizing column headers.
- Encoding categorical variables (e.g., Employment_Status) using Label Encoding.
- Verifying the dataset for missing values; none were found.
- Normalizing numerical features using StandardScaler to ensure uniform scale.
- Splitting the dataset into training and test sets in an 80:20 ratio.
- Applying SMOTE (Synthetic Minority Over-sampling Technique) to the training set to address class imbalance.

## Model Development

Three machine learning algorithms were developed and assessed:

1. *Support Vector Machine (SVM)*  
   Utilized with a radial basis function (RBF) kernel and class_weight parameter set to 'balanced'. Hyperparameter tuning was performed using GridSearchCV. The tuned SVM achieved an accuracy of 97.25%, with a precision of 0.96 and recall of 0.90 for the approved loan class.

2. *Decision Tree Classifier*  
   Implemented with class balancing and default hyperparameters. The Decision Tree achieved perfect accuracy on the test set (100%). However, this may indicate potential overfitting, warranting caution in deployment scenarios.

3. *Random Forest Classifier*  
   Similar to the Decision Tree, the Random Forest also reported 100% accuracy. Due to its ensemble approach, it provides improved generalizability compared to a single tree, although identical test results suggest model performance should be further validated.

## Evaluation Metrics

Model performance was assessed using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC score

Although all models performed strongly, the SVM demonstrated better generalization with slightly lower but more realistic accuracy compared to the tree-based models. All models achieved ROC-AUC scores of 1.00, indicating excellent discriminatory power.

## Feature Importance

An analysis of feature importance revealed that Credit Score, Loan Amount, and Income were the most influential predictors in all three models. Employment Status, Loan Term, and Age were found to have lesser impact on the model’s decision boundary. Feature importance was derived from both coefficient analysis (for SVM) and Gini-based importance scores (for tree-based models).

## Ethical Considerations

- The dataset is fully anonymized and contains no personally identifiable information (PII).
- It is compliant with GDPR and University of Hertfordshire ethical guidelines.
- Ethical approval was not required as no human participants or primary data collection were involved.

## Repository Structure

This GitHub repository is organized as follows:

- data/ – Contains the loan dataset.
- src/ – Scripts for preprocessing, model training, and evaluation.
- notebooks/ – Jupyter notebooks for EDA and model development.
- figures/ – Visual outputs, including ROC curves and confusion matrices.
- README.md – Project description and documentation.
- requirements.txt – List of dependencies.

## Conclusion and Future Work

The results demonstrate the strong potential of machine learning algorithms in automating and improving loan approval decisions. While Decision Tree and Random Forest models exhibited perfect accuracy, the SVM provided a more balanced performance profile suitable for real-world applications. Future enhancements may include the integration of explainable AI (XAI) tools such as SHAP, model deployment in financial software systems, and testing on larger, real-world datasets.

## Author and Supervision

- *Student Name*: Prathish Kumar Peddakotla  
- *SRN*: 23026596  
- *Supervisor*: Sunina Sharya  
- *Course*: MSc Data Science, University of Hertfordshire  
- *Date Submitted*: 29 April 2025
