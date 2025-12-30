# 🩺 Disease Prediction from Medical Data - Autism Spectrum Disorder (ASD) Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset Information](#dataset-information)
- [Technologies & Tools](#technologies--tools)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Models Implemented](#models-implemented)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results & Performance](#results--performance)
- [Visualizations](#visualizations)
- [Key Features](#key-features)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

This project implements a comprehensive **Machine Learning solution for Autism Spectrum Disorder (ASD) prediction** using medical data. The system leverages multiple ML algorithms to provide accurate early detection of autism, enabling timely intervention and support for individuals. 

**Autism** is a neurodevelopmental disorder characterized by: 
- Challenges in social interaction
- Communication difficulties
- Restricted and repetitive patterns of behavior and interests

Early detection is crucial for initiating appropriate interventions and improving the quality of life for individuals with ASD.

## 🩹 Problem Statement

The primary challenge addressed in this project is the **need for accurate and timely autism prediction**, particularly in early developmental stages. Given a dataset of 1000 individuals who completed an app-based screening form, this project aims to:

1. Build reliable machine learning models for autism prediction
2. Improve early identification of ASD symptoms
3. Enable timely intervention and support for individuals with ASD
4. Handle imbalanced dataset (80% negative, 20% positive cases)

## 📊 Dataset Information

### Dataset Size
- **Training Set**: 800 samples (`train.csv`)
- **Test Set**: 200 samples (`test.csv`)

### Features (22 Columns)

#### Screening Scores
- `A1_Score` to `A10_Score` - Scores based on Autism Spectrum Quotient (AQ) 10-item screening tool

#### Demographic Information
- `ID` - Patient identification number
- `age` - Age of the patient (in years)
- `gender` - Gender of the patient (Male/Female)
- `ethnicity` - Ethnic background
- `contry_of_res` - Country of residence (56 unique countries)

#### Medical History
- `jaundice` - Whether the patient had jaundice at birth (Yes/No)
- `austim` - Family history of autism (Yes/No)

#### Assessment Details
- `used_app_before` - Previous screening test history (Yes/No)
- `result` - Cumulative score for AQ1-10 screening test
- `age_desc` - Age category of the patient
- `relation` - Relationship of person completing the test (Self/Parent/Relative/Healthcare Professional)

#### Target Variable
- `Class/ASD` - Binary classification (0 = No ASD, 1 = ASD detected)

### Data Characteristics
- **Class Imbalance**: 79.87% negative class (639 samples), 20.13% positive class (161 samples)
- **No Missing Values**: Complete dataset with no null values
- **Categorical Features**: 7 features (gender, ethnicity, jaundice, autism family history, country, app usage, relation)
- **Numerical Features**: 13 features (10 AQ scores + age + result + target)

## 🛠️ Technologies & Tools

### Programming Language
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) **Python 3.8+**

### Core Libraries

#### Data Manipulation & Analysis
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) **Pandas** - Data manipulation and analysis
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) **NumPy** - Numerical computing

#### Data Visualization
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) **Matplotlib** - Static visualizations and plots
- ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat) **Seaborn** - Statistical data visualization

#### Machine Learning
- ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) **Scikit-learn** - ML algorithms and preprocessing
  - Classification Models
  - Preprocessing (OrdinalEncoder, StandardScaler)
  - Model Selection (train_test_split)
  - Metrics (accuracy_score, classification_report, confusion_matrix)

### Development Environment
- ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) **Jupyter Notebook** - Interactive development and documentation

### Data Encoding
- **OrdinalEncoder** - Categorical feature encoding with unknown value handling

## 🔄 Machine Learning Pipeline

### 1. **Data Preprocessing**
```
├── Data Loading (train.csv, test.csv)
├── Exploratory Data Analysis (EDA)
├── Feature Engineering
│   ├── Remove ID column (non-informative)
│   ├── Remove age_desc (zero variance)
│   ├── Handle missing values ('?' symbols)
│   ├── Merge duplicate categories (Others/others)
├── Categorical Encoding
│   └── OrdinalEncoder (7 categorical features)
└── Class Imbalance Handling
```

### 2. **Feature Selection**
- **Initial Features**: 22 columns
- **After Preprocessing**: 20 features (removed ID, age_desc)
- **Categorical Features Encoded**: gender, ethnicity, jaundice, austim, contry_of_res, used_app_before, relation

### 3. **Model Training & Evaluation**
```
├── Train-Test Split
├── Model Training (Multiple Algorithms)
├── Hyperparameter Tuning
├── Cross-Validation
└── Performance Evaluation
```

## 🤖 Models Implemented

The project implements and compares multiple machine learning algorithms:

1. **Logistic Regression**
   - Binary classification baseline model
   - Linear decision boundary

2. **Decision Tree Classifier**
   - Non-linear model
   - Interpretable decision rules

3. **Random Forest Classifier**
   - Ensemble learning method
   - Reduces overfitting

4. **Support Vector Machine (SVM)**
   - Effective for high-dimensional data
   - Kernel-based classification

5. **K-Nearest Neighbors (KNN)**
   - Instance-based learning
   - Non-parametric approach

6. **Gradient Boosting Classifier**
   - Sequential ensemble method
   - High accuracy potential

7. **XGBoost / LightGBM** (if implemented)
   - Advanced gradient boosting
   - Optimized performance

## 📁 Project Structure

```
Disease_Prediction/
│
├── Autism_Prediction.ipynb    # Main Jupyter notebook with complete implementation
├── train.csv                   # Training dataset (800 samples)
├── test.csv                    # Test dataset (200 samples)
├── README.md                   # Project documentation (this file)
│
├── data/                       # (Optional) Additional data files
├── models/                     # (Optional) Saved model files
├── visualizations/             # (Optional) Generated charts and graphs
└── requirements.txt            # (Optional) Python dependencies
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Step 1: Clone the Repository
```bash
git clone https://github.com/Gouravlamba/Disease_Prediction.git
cd Disease_Prediction
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Or create a `requirements.txt`:
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook
```bash
jupyter notebook
```

Open `Autism_Prediction.ipynb` in your browser.

## 💻 Usage

### Running the Complete Pipeline

1. **Open the Notebook**
   ```bash
   jupyter notebook Autism_Prediction.ipynb
   ```

2. **Execute Cells Sequentially**
   - Run all cells from top to bottom
   - Or use "Run All" from the Cell menu

3. **Key Sections to Focus On**
   - Data Preprocessing
   - Exploratory Data Analysis (EDA)
   - Model Training
   - Performance Evaluation
   - Predictions on Test Set

### Making Predictions on New Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Load your trained model (example)
# model = joblib.load('autism_prediction_model.pkl')

# Prepare new data
new_data = pd.DataFrame({
    'A1_Score': [1], 'A2_Score': [1], 'A3_Score': [1],
    # ... (include all features)
})

# Preprocess and predict
# prediction = model.predict(new_data)
```

## 📈 Results & Performance

### Model Comparison
*(Results will be added based on your actual model performance)*

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | XX% | XX% | XX% | XX% |
| Decision Tree | XX% | XX% | XX% | XX% |
| Random Forest | XX% | XX% | XX% | XX% |
| SVM | XX% | XX% | XX% | XX% |
| KNN | XX% | XX% | XX% | XX% |
| Gradient Boosting | XX% | XX% | XX% | XX% |

### Best Performing Model
🏆 **[Model Name]** achieved the highest accuracy of **XX%** on the test set.

### Handling Class Imbalance
- **Technique**: SMOTE / Class Weights / Undersampling
- **Impact**: Improved recall for positive class by XX%

## 📊 Visualizations

The notebook includes comprehensive visualizations:

### 1. **Exploratory Data Analysis**
- Distribution of target variable (Class/ASD)
- Age distribution by autism diagnosis
- Gender distribution across classes
- Country-wise autism prevalence
- Ethnicity distribution

### 2. **Feature Analysis**
- Correlation heatmap of AQ scores
- Box plots for numerical features
- Count plots for categorical features
- Screening score distributions (A1-A10)

### 3. **Model Performance**
- Confusion matrices for all models
- ROC curves and AUC scores
- Precision-Recall curves
- Feature importance plots (for tree-based models)
- Learning curves

### 4. **Comparative Analysis**
- Model accuracy comparison bar chart
- F1-score comparison
- Training vs validation performance

### Sample Visualizations

```python
# Example: Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Autism Prediction')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

```python
# Example: Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Importance Score')
plt.title('Feature Importance in Autism Prediction')
plt.tight_layout()
plt.show()
```

## ✨ Key Features

✅ **Comprehensive Data Preprocessing**
- Automatic handling of missing values
- Intelligent categorical encoding
- Feature engineering

✅ **Multiple ML Algorithms**
- Comparison of 6+ different models
- Ensemble methods implementation

✅ **Class Imbalance Handling**
- Techniques to handle 80-20 class distribution

✅ **Robust Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix analysis
- Cross-validation

✅ **Interactive Jupyter Notebook**
- Step-by-step documentation
- Visual insights at each stage

✅ **Scalable Architecture**
- Easy to add new features
- Modular code structure

## 🔮 Future Enhancements

- [ ] **Deep Learning Implementation**
  - Neural Networks (ANN)
  - LSTM for sequential data

- [ ] **Web Application Development**
  - Flask/Django API
  - React/Streamlit frontend
  - Real-time prediction interface

- [ ] **Model Deployment**
  - Docker containerization
  - Cloud deployment (AWS/Azure/GCP)
  - REST API for predictions

- [ ] **Advanced Feature Engineering**
  - Polynomial features
  - Interaction terms
  - PCA for dimensionality reduction

- [ ] **Explainable AI (XAI)**
  - SHAP values
  - LIME explanations
  - Model interpretability

- [ ] **Mobile Application**
  - Flutter/React Native app
  - Offline prediction capability

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include unit tests for new features
- Update documentation

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Contact

**Gourav Lamba**

- GitHub: [@Gouravlamba](https://github.com/Gouravlamba)
- Project Link: [https://github.com/Gouravlamba/Disease_Prediction](https://github.com/Gouravlamba/Disease_Prediction)

---

## 🙏 Acknowledgments

- Dataset source: Autism Spectrum Quotient (AQ) screening data
- Scikit-learn documentation and community
- Jupyter Notebook development team
- Open-source ML community

---

## 📚 References

1. Autism Spectrum Quotient (AQ) screening methodology
2. Scikit-learn documentation: https://scikit-learn.org/
3. WHO - Autism Spectrum Disorders: https://www.who.int/news-room/fact-sheets/detail/autism-spectrum-disorders
4. Machine Learning for Healthcare applications

---

<div align="center">

### ⭐ If you find this project useful, please consider giving it a star! ⭐

**Made with ❤️ by Gourav Kumar**

</div>
