# **Machine Learning Assignment: Regression and Classification**

## **Overview**

This repository contains the complete code, datasets, and documentation for a hands-on machine learning assignment. The focus was to explore and implement regression and classification techniques on real-world datasets. Key tasks included:

1. **Predicting House Prices** using the Ames Housing dataset with regression models.
2. **Customer Churn Prediction** using logistic regression for binary classification.

---

## **Project Structure**

```
main_dir
├── datasets
│   ├── AmesHousing (raw)
│   ├── without-na
│   ├── without-ols
│   ├── encoded-unscaled
│   ├── scaled
│   └── most-imp-39-features
│   ├── Customer Churn
│       ├── telco-customer-churn (raw)
│       └── encoded_unscaled
├── notebooks
│   ├── preprocessing
│   │   ├── analysis (ames-housing)
│   │   ├── encoding (ames-housing)
│   │   └── telco_preparation
│   └── model_building
│       ├── 1-linear-regression
│       └── 2-2-logistic-regression
│       └── 3-2.1-multinomial-logistic-regression
├── requirements.txt
├── encoding.txt
└── data.json
```

---

## **Tasks and Methodology**

### **1. Regression Task: Predicting House Prices**
- Dataset: Ames Housing
- Techniques:
  - Simple Linear Regression
  - Multiple Linear Regression
  - Polynomial Regression
- Preprocessing Steps:
  - Handling missing values
  - Removing outliers
  - Feature encoding and scaling
  - Correlation analysis for feature selection
- Results:
  - **RMSE**: ~18,000
  - **R² Score**: 0.89

### **2. Classification Task: Predicting Customer Churn**
- Dataset: Customer Churn
- Techniques:
  - Logistic Regression (binary classification)
  - Multinomial Logistic Regression (exploratory)
- Preprocessing Steps:
  - Feature encoding
  - Dataset balancing (not implemented due to time constraints)
- Results:
  - **Accuracy**: ~79%

---

## **Setup Instructions**

1. Clone this repository:
   ```bash
   git clone [repository-url]
   ```

2. Navigate to the project directory:
   ```bash
   cd 1 (main_dir)
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Open the notebooks to explore the preprocessing and model-building steps:
   ```bash
   jupyter notebook
   ```

---

## **Key Insights**

- The importance of data preprocessing in improving model performance.
- Using correlation analysis for feature selection in high-dimensional datasets.
- Visualizations to uncover patterns and insights in classification tasks.

---

## **Resources**

- **Dataset Links:**
  - [Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)
- **Blog Post:** [Detailed Project Walkthrough](https://shaukat.tech/assignment-3-implementing-linear-and-logistic-regression/)

---

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.

---

## **Acknowledgements**

Special thanks to the **Innoquest Cohort-1 Machine Learning Module** for the assignment and resources provided.

