# Customer Happiness Prediction Project

## 📌 Introduction  
The **Customer Happiness Prediction** project aims to use machine learning techniques to predict customer satisfaction based on various features. By analyzing historical data, this project applies different classification algorithms to build a predictive model that classifies customers into two categories: **happy** or **unhappy**.

This project demonstrates how feature selection, model training, and evaluation techniques can be applied to real-world data to generate meaningful business insights.

---

## 🎯 Objectives  

- Predict customer happiness based on several factors such as delivery time, order accuracy, and pricing.  
- Implement a variety of classification algorithms (Logistic Regression, Decision Trees, LightGBM, etc.).  
- Explore hyperparameter optimization to improve model performance.  
- Visualize the results and evaluate model performance using metrics like accuracy, confusion matrix, and classification report.

---

## 📊 Data Overview  

The dataset used in this project includes customer feedback, with features such as:  

- Delivered on Time  
- Expected Order Delivery  
- Good Price for Order  
- **Happiness** (Target variable)

The target variable, `Happiness`, is a binary classification label where:  

- `0` = Unhappy  
- `1` = Happy  

The dataset is split into training and test sets to evaluate model performance.

---

## ⚙️ Steps and Methodology

### 🔹 Data Preprocessing  

- Loaded the dataset and performed initial analysis to check for missing or erroneous data.  
- Split the data into training and testing sets using `train_test_split`.  
- Applied **Recursive Feature Elimination (RFE)** to select the most important features.

### 🔹 Model Training  

Trained multiple machine learning models, including:  

- Logistic Regression  
- Decision Tree Classifier  
- LightGBM  
- AdaBoost  
- Bagging Classifier  
- Voting Classifier (Hard & Soft)

Hyperparameter tuning was performed using the **Hyperopt** library to find the best-performing parameters for the **LightGBM** model.

### 🔹 Model Evaluation  

Evaluated model performance using:  

- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-Score)

Visualized the performance using **confusion matrix heatmaps**.

---

## 🔍 Feature Selection and Optimization  

- Used **Recursive Feature Elimination (RFE)** to reduce the feature set to the most relevant ones.  
- Optimized **LightGBM** hyperparameters using **Hyperopt**, leading to improved model accuracy and generalization.

---

## ✅ Conclusion  

- After training multiple models and tuning hyperparameters, **LightGBM** with optimal settings delivered the best performance.  
- Feature selection helped improve model performance by reducing overfitting.  
- The final model achieved strong classification accuracy, highlighting key factors like delivery time and pricing as significant predictors of customer happiness.

---

## 📈 Key Findings and Conclusion  

- **Best Performing Model**: LightGBM, after hyperparameter tuning, outperformed other models in terms of accuracy and generalization.  
- **Feature Importance**: Factors like delivery timing and order pricing were identified as strong predictors.  
- **Model Evaluation**: Confusion matrix and classification report confirmed high accuracy in distinguishing happy vs. unhappy customers.

---

## 💻 Installation and Usage  

To run this project locally:

```bash
git clone https://github.com/codebrew09/Machine-Learning.git
cd Machine-Learning
pip install -r requirements.txt
```

### 🧪 Running the notebook  

- Open `Customer_Happiness_Project_Final.ipynb` using **Jupyter Notebook** or any compatible notebook editor.  
- Run the cells sequentially to execute the code and view results.

---

## 🏆 Accomplishments  

- Successfully implemented machine learning models to predict customer happiness.  
- Achieved high accuracy through hyperparameter optimization and feature selection.  
- Created insightful visualizations to aid in understanding model behavior and predictions.

---

## 🚀 Future Work  

- **Explore More Models**: Try models like Random Forest, XGBoost, or deep learning for comparison.  
- **Real-Time Application**: Integrate the model into a live feedback system for real-time predictions.  
- **Feature Engineering**: Incorporate additional features such as customer demographics or order history to enhance predictive power.
