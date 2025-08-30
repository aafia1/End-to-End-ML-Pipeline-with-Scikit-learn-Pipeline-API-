# Customer Churn Prediction – End-to-End ML Pipeline

This repository contains an end-to-end **Machine Learning pipeline** for predicting customer churn using the **Telco Churn Dataset**. The project follows production-ready practices with preprocessing, model training, evaluation, and pipeline export.

---

## 📌 Internship Context
This project was developed as part of  
**DevelopersHub Corporation**  
**AI/ML Engineering – Advanced Internship Tasks**  

**Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API**  

---

## 🚀 Project Overview
- Implemented data preprocessing (scaling, encoding) using **Scikit-learn Pipeline API**  
- Trained **Logistic Regression** and **Random Forest** models  
- Performed **hyperparameter tuning** using `GridSearchCV`  
- Exported the complete pipeline using **Joblib** for reusability  
- Evaluated model with metrics like **Accuracy**, **F1-Score**, and **Confusion Matrix**

---

## 📂 Folder Structure
churn_pipeline_project/
│
├── churn_train.ipynb        # Main Jupyter Notebook
├── requirements.txt         # Dependencies
│
├── data/
│   └── raw/
│       └── telco.csv        # Dataset
│
├── models/
│   └── best_model.pkl       # Exported trained model
│
└── metrics.json             # Evaluation results

---

## ⚙️ Setup Instructions
**1️⃣ Clone the repository**
git clone https://github.com/aafia1/churn_pipeline_project.git
cd churn_pipeline_project

**2️⃣ Create environment & install dependencies**
conda create -n churn-ml python=3.10 -y
conda activate churn-ml
pip install -r requirements.txt

**3️⃣ Run the Jupyter Notebook**
jupyter notebook churn_train.ipynb

---

## 📊 Results
- The Random Forest model achieved the best performance with tuned hyperparameters.
- The final pipeline is exported as:
  models/best_model.pkl → serialized model for production use
  metrics.json → evaluation results

---

## 🛠️ Skills Gained
- ML pipeline construction with Scikit-learn
- Hyperparameter tuning with GridSearchCV
- Model export & reusability with Joblib
- Building production-ready pipelines

## 👨‍💻 Contributors
Developed by [Aafia Azhar]
as part of DevelopersHub Corporation – AI/ML Engineering Advanced Internship
