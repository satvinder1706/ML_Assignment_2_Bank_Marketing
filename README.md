# Bank Marketing Subscription Prediction
## M.Tech (Data Science & Engineering) — Machine Learning Assignment 2

## A. Problem statement

The objective of this project is to develop and compare five supervised machine-learning classification models to predict whether a bank customer will subscribe to a term deposit (`yes` or `no`) following a direct-marketing campaign. The project also provides an interactive Streamlit application through which a user can select a trained model, upload labelled test data, and inspect evaluation metrics, a confusion matrix, a classification report, and sample predictions.

This is a binary classification problem. The positive class evaluated in the metrics is **`yes` (subscribed)**.

## B. Dataset description

- **Dataset:** Bank Marketing
- **Source:** UCI Machine Learning Repository
- **Dataset ID:** 222
- **Official dataset page:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Original observations:** 45,211 customer records
- **Original input variables:** 16
- **Predictor variables used in this experiment:** 15
- **Target:** `y` — whether the customer subscribed to a term deposit (`yes`/`no`)

Each instance represents a customer record. The predictors describe customer demographics, financial information and campaign-contact history. The target column records whether the customer subscribed to the term deposit.

The `duration` predictor was excluded because call duration is only known after the marketing call has taken place. Including it in a model intended to support decisions before or during customer targeting could introduce target leakage.

**Dataset reference:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

## C. GitHub repository link

**GitHub repository:** `[PASTE YOUR PUBLIC GITHUB REPOSITORY URL HERE]`

The repository should contain the complete source code, `requirements.txt`, this `README.md`, the labelled test data (`test_data.csv`), and the saved model files in the `model/` directory.

## D. Models used and evaluation

The following five classification algorithms were trained and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Experimental setup

- **Train/test split:** 80% training and 20% testing.
- **Split method:** Stratified train/test split.
- **Random state:** 42.
- **Test-set size:** 9,043 records.
- **Test-set class distribution:** 7,985 `no` records and 1,058 `yes` records.
- **Numeric preprocessing:** Median imputation followed by standard scaling.
- **Categorical preprocessing:** Most-frequent imputation followed by one-hot encoding.
- **Model evaluation:** Accuracy, ROC-AUC, precision, recall, F1-score and Matthews Correlation Coefficient (MCC).

All metrics below are the recorded results from the experiment. Precision, recall and F1-score are for the positive (`yes`) class. AUC is calculated using the model's probability estimate or decision score.

### Comparison of evaluation metrics

| ML model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | **0.7722** | 0.2662 | **0.6238** | 0.3732 | 0.2853 |
| Decision Tree | 0.8251 | 0.7517 | 0.3452 | 0.5520 | **0.4247** | **0.3406** |
| K-Nearest Neighbors | **0.8891** | 0.7138 | **0.5758** | 0.1975 | 0.2942 | 0.2919 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.7578 | 0.7718 | 0.2684 | 0.6200 | 0.3746 | 0.2867 |

### Observations on model performance

#### 1. Logistic Regression

Logistic Regression achieved an accuracy of 0.7548 and the highest AUC of 0.7722 among the five models. It also achieved the highest recall (0.6238), meaning it identified the largest proportion of actual term-deposit subscribers in this experiment. Its precision was 0.2662, however, indicating that a substantial proportion of customers predicted as subscribers were not actual subscribers. This model may be useful when ranking or identifying a broader pool of potential subscribers is more important than limiting false positives.

#### 2. Decision Tree Classifier

The Decision Tree achieved an accuracy of 0.8251, an F1-score of 0.4247 and an MCC of 0.3406. It recorded the highest F1-score and MCC among the five models, indicating the strongest overall balance of positive-class precision and recall according to these measures. Its precision was 0.3452 and recall was 0.5520. On the basis of the reported metrics, the Decision Tree is the strongest balanced performer in this experiment.

#### 3. K-Nearest Neighbors (KNN)

KNN achieved the highest accuracy (0.8891) and the highest precision (0.5758). However, its recall was only 0.1975 and its F1-score was 0.2942. This means that it correctly identified a relatively small proportion of the actual subscribers, even though its overall accuracy was high. Its AUC of 0.7138 was also the lowest among the five models. The result illustrates why accuracy alone is not sufficient for judging performance on an imbalanced classification dataset.

#### 4. Gaussian Naive Bayes

Naive Bayes achieved an accuracy of 0.8452, precision of 0.3708, recall of 0.4641, F1-score of 0.4123 and MCC of 0.3271. Its F1-score and MCC were the second-highest after the Decision Tree. It therefore provided a reasonably balanced result, although its AUC of 0.7514 was below the AUC values of Logistic Regression and SVM.

#### 5. Support Vector Machine (SVM)

The SVM achieved an accuracy of 0.7578 and an AUC of 0.7718, which was close to the Logistic Regression AUC. Its recall was 0.6200, also close to Logistic Regression, while its precision was 0.2684. Its F1-score was 0.3746 and MCC was 0.2867. The model showed good ability to identify subscribers relative to several alternatives, but it generated a comparatively high number of false-positive predictions.

### Overall winner for this dataset

**Decision Tree Classifier is selected as the overall winner when F1-score and MCC are used to assess balanced positive-class performance.** It recorded the highest F1-score (0.4247) and highest MCC (0.3406). These measures are more informative than accuracy alone when the target classes are imbalanced.

The best model can nevertheless depend on the bank's objective. If the priority is to identify as many actual subscribers as possible, Logistic Regression may be preferred because it achieved the highest recall (0.6238) and AUC (0.7722). If reducing false-positive predictions is more important, KNN had the highest precision (0.5758), but its low recall means it missed many actual subscribers.

### Class imbalance and interpretation

The test set contained 7,985 non-subscribers and 1,058 subscribers. Thus, approximately 88.3% of the test records belonged to the `no` class. This class imbalance helps explain why a model can achieve high accuracy while performing less effectively at identifying subscribers. For this reason, the comparison considers AUC, precision, recall, F1-score and MCC in addition to accuracy.

## E. Streamlit application

The Streamlit application provides the following features:

- Upload of labelled test data in CSV format.
- Dropdown selection of one of the five trained classification models.
- Display of accuracy, AUC, precision, recall, F1-score and MCC.
- Confusion matrix.
- Classification report.
- Preview of actual and predicted labels.
- Download of the prediction preview.

**Live Streamlit application:** `[PASTE YOUR DEPLOYED STREAMLIT APP URL HERE]`

The application expects the target column in the uploaded labelled test CSV to be named `y` and the predictor columns to match the training dataset.

## F. How to run the project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the models and generate the evaluation results and test file:

```bash
python train.py
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

The training script generates `metrics.csv`, `test_data.csv`, training metadata and the saved model files under `model/`.

## G. Repository contents

```text
project-folder/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── metrics.csv
├── test_data.csv
├── training_metadata.json
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knearest_neighbors.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── label_classes.json
```

## H. Execution evidence and submission details

- **BITS Virtual Lab execution screenshot:** `[INSERT YOUR REQUIRED SCREENSHOT IN THE SUBMISSION PDF]`
- **GitHub repository:** `[PASTE LINK]`
- **Live Streamlit app:** `[PASTE LINK]`

Before final submission, verify that the GitHub repository is accessible, the deployed Streamlit application opens and runs, the required files are present, and the BITS Virtual Lab screenshot is included in the single PDF submission.

---

*This README reports the metrics generated by the experiment. The repository and live application URLs and the BITS Virtual Lab screenshot must be filled in by the student before submission.*
