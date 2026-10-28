Breast Cancer Diagnosis Classifier
A binary classification project that predicts whether a breast tumour is malignant or benign from 30 numeric features, using Logistic Regression.

Dataset
Source: load_breast_cancer() from scikit-learn (Breast Cancer Wisconsin Diagnostic dataset)
30 numeric features, no missing values, no duplicates
Target: 0 = malignant, 1 = benign
Workflow
Exploration: checked missing values and duplicates, ran describe(), and drew boxplots for all 30 features.
Outlier removal: removed rows outside 1.5 x IQR, which left 398 records (300 benign, 98 malignant).
Class balancing: applied SMOTE (imbalanced-learn) to reach 300 / 300 samples.
Feature selection: kept the 10 best features with SelectKBest (ANOVA F-test).
Training: 80/20 train-test split (random_state=42) and a Logistic Regression model.
Evaluation: accuracy, confusion matrix, and classification report.
Selected features
mean radius, mean perimeter, mean area, mean concavity, mean concave points, worst radius, worst perimeter, worst area, worst concavity, worst concave points.

Results
<!-- UPDATE these numbers after you re-run the notebook with SMOTE applied after the split -->
Metric	Value
Train accuracy	94.2%
Test accuracy	95.8% (120 test samples)
Malignant (0) precision / recall	0.94 / 0.98
Benign (1) precision / recall	0.98 / 0.93
Limitations
SMOTE was applied before the train-test split, so the test set contains synthetic samples and the accuracy may be optimistic. Applying SMOTE only on the training set is the planned fix.
IQR outlier removal dropped a large share of the original records.
Logistic Regression showed a convergence warning; feature scaling or a higher max_iter should be tested.
Tools
Python, Pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib, Seaborn

Files
sklearn_breast_cancer_dataset.ipynb: full notebook
Run it
bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook sklearn_breast_cancer_dataset.ipynb
Claude finished the response
