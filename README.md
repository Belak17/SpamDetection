#  SMS Spam Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NLTK](https://img.shields.io/badge/NLTK-Natural%20Language%20Processing-154F8B?style=flat)](https://www.nltk.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

> A Natural Language Processing (NLP) project that detects spam SMS messages using Machine Learning, text preprocessing, feature engineering, and classification algorithms.

---

#  Table of Contents

- [About](#about)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Data Processing](#data-processing)
- [NLP Pipeline](#nlp-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Screenshots](#screenshots)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Saving](#model-saving)
- [Future Improvements](#future-improvements)
- [Author](#author)


---

# About

SMS Spam Detection is a Machine Learning project designed to classify SMS messages into two categories:

- **Spam** → unwanted or fraudulent messages
- **Ham** → legitimate messages

The project implements a complete NLP workflow:

- Data cleaning
- Feature engineering
- Text preprocessing
- Text vectorization
- Model training
- Model comparison
- Evaluation
- Prediction on new messages

---

# Objectives

The main objectives of this project are:

- Build an automatic SMS spam classifier
- Apply Natural Language Processing techniques
- Transform text data into numerical features
- Compare different Machine Learning algorithms
- Handle class imbalance
- Select the best performing model

---

# Dataset

The dataset contains SMS messages with their labels:

| Label | Meaning |
|---|---|
| 0 | Ham (Legitimate Message) |
| 1 | Spam |

Additional features:

- Message text
- URL information
- Email information
- Phone number information
- Language information

---

# Data Processing

## Data Cleaning

Applied preprocessing steps:

- Remove unnecessary columns
- Handle missing values
- Encode labels
- Normalize language information

---

# Feature Engineering

Created additional features:

- Word count
- URL detection
- Email detection
- Phone number detection
- Number detection

Regex replacements:

```
Email       → emailaddr
URL         → httpaddr
Money       → moneysymb
Phone       → phonenumbr
Numbers     → numbr
```

---

# NLP Pipeline

The text preprocessing workflow:

```
Raw SMS Text

        ↓

Lowercase Conversion

        ↓

Regex Cleaning

        ↓

Tokenization

        ↓

Stopword Removal

        ↓

Porter Stemming

        ↓

CountVectorizer

        ↓

Machine Learning Model

        ↓

Spam / Ham Prediction
```

---

# Machine Learning Models

The following models were tested:

- Logistic Regression
- Logistic Regression with class balancing
- Linear SVM
- Linear SVM with class balancing
- Decision Tree Classifier


Text representation:

```
CountVectorizer
```

Features:

```
3000 text features + Word_Count
```

---

# Model Comparison

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score


| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression Balanced | **98.20%** | 96.31% | **94.76%** | **95.53%** |
| Linear SVM Balanced | 97.88% | 97.03% | 92.34% | 94.63% |
| Logistic Regression | 97.80% | 97.02% | 91.94% | 94.41% |
| Linear SVM | 97.63% | 96.60% | 91.53% | 93.99% |
| Decision Tree | 95.92% | 91.25% | 88.31% | 89.75% |

---

# Best Model

The selected model:

```python
LogisticRegression(class_weight="balanced")
```

Performance:

```
Accuracy : 98.20%

Spam Precision : 96.31%

Spam Recall : 94.76%

Spam F1-score : 95.53%
```

Using class balancing improved spam detection by giving more importance to the minority class.

---

---

# Screenshots

## Dataset Overview

Preview of the dataset after loading and initial analysis:

![Dataset Overview](/images/dataset_preview.png)

---

## Data Distribution

Distribution of SMS messages between Spam and Ham classes:

![Class Distribution](/images/class_distribution.png)

---

## Model Comparison

Comparison of Machine Learning models based on Accuracy, Precision, Recall, and F1-score:

![Model Comparison](/images/model_comparison.png)

---

## Spam Prediction Example

Example of a prediction performed by the trained model:

![Prediction Example](/images/prediction_example.png)

---

# Prediction Example

Example input:

```
Congratulations! You won a free prize. Call now!
```

Prediction:

```
Spam
```

Example input:

```
Hey, are we meeting tomorrow?
```

Prediction:

```
Ham
```

---

# Model Saving

The trained components are saved using `joblib`.

Structure:

```
models/

├── spam_model.pkl
├── count_vectorizer.pkl
└── features_columns.pkl
```

Saved elements:

- Trained classifier
- CountVectorizer vocabulary
- Feature ordering

This allows prediction without retraining.

---

# Technologies

## Programming Language

- Python

## Machine Learning

- Scikit-Learn
- Pandas
- NumPy

## NLP

- NLTK
- Regex

## Tools

- Jupyter Notebook
- VS Code
- PyCharm

---

# Project Structure

```
SMS-Spam-Detection/

│
├── dataset/
│   └── sms_dataset.csv
│
├── notebooks/
│   └── spam_detection.ipynb
│
├── models/
│   ├── spam_model.pkl
│   ├── count_vectorizer.pkl
│   └── features_columns.pkl
│
├── images/
│   ├── dataset_preview.png
│   ├── class_distribution.png
│   ├── model_comparison.png
│   └── prediction_example.png
│
├── requirements.txt
│
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Belak17/<repository-name>.git

cd <repository-name>
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Notebook

```bash
jupyter notebook
```

Execute the notebook to reproduce:

- Data preprocessing
- Model training
- Evaluation
- Prediction

---

# Future Improvements

Possible improvements:

- TF-IDF Vectorization
- Word Embeddings
- Word2Vec / GloVe
- Transformer Models (BERT)
- Hyperparameter Optimization
- Cross Validation
- FastAPI REST API
- Streamlit Interface
- Docker Deployment

---

# Author

## Kaleb AKAKPO

Software Engineering Student  
Machine Learning Enthusiast

GitHub:

https://github.com/Belak17


---

<div align="center">

 If you like this project, consider giving it a star on GitHub!

</div>