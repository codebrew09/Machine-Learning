Customer Happiness Prediction Project

INTRODUCTION
The "Customer Happiness Prediction" project aims to use machine learning techniques to predict customer happiness based on various features. By analyzing historical data, the project applies different classification algorithms to build a predictive model that classifies customer satisfaction into two categories: happy or unhappy. The project demonstrates how feature selection, model training, and evaluation techniques can be applied to real-world data for business insights.

OBJECTIVES
•	Predict customer happiness based on several factors such as delivery time, order accuracy, and pricing.
•	Implement a variety of classification algorithms (Logistic Regression, Decision Trees, LightGBM, etc.).
•	Explore hyperparameter optimization to improve model performance.
•	Visualize the results and evaluate the performance using metrics like accuracy, confusion matrix, and classification report.

DATA OVERVIEW
The dataset used in this project includes customer feedback, including features such as:

•	Delivered on Time
•	Expected Order Delivery
•	Good Price for Order
•	Happiness (Target variable)

The target variable, Happiness, is a binary classification label where 0 represents "Unhappy" and 1 represents "Happy." The dataset is split into a training and test set to train and evaluate the model performance.

STEPS AND METHODOLOGY
1. Data Preprocessing
•	Loaded the dataset and performed initial analysis to check for any missing or erroneous data.
•	Split the data into training and testing sets using train_test_split.
•	Applied Recursive Feature Elimination (RFE) to select important features.

2. Model Training
•	Trained multiple machine learning models, including:

  o	 Logistic Regression
  o	Decision Tree Classifier
  o	LightGBM
  o	AdaBoost
  o	Bagging Classifier
  o	Voting Classifier (Hard & Soft)

•	Hyperparameter tuning was performed using the Hyperopt library to find the best-performing parameters for the LightGBM model.

3. Model Evaluation
•	Evaluated model performance using various metrics:
  o	Accuracy Score
  o	Confusion Matrix
  o	Classification Report (Precision, Recall, F1-Score)

•	Visualized the performance of each model using confusion matrix heatmaps.

4. Feature Selection and Optimization
•	Used Recursive Feature Elimination (RFE) to reduce the feature set to the most relevant ones.
•	Optimized the hyperparameters of LightGBM using Hyperopt to improve model accuracy.

5. Conclusion
•	After training multiple models and tuning hyperparameters, LightGBM with optimal settings delivered the best performance.
•	Feature selection helped improve model performance by reducing overfitting.
•	The final model showed a promising classification accuracy, indicating that key factors such as delivery time and pricing significantly affect customer happiness.

KEY FINDINGS AND CONCLUSION
•	Best Performing Model: LightGBM, after hyperparameter tuning, outperformed other models in terms of accuracy and generalization.
•	Feature Importance: Factors like delivery timing and order pricing were identified as significant predictors of customer happiness.
•	Model Evaluation: The confusion matrix and classification report showed that the models, especially LightGBM, classified happy vs. unhappy customers with a high degree of accuracy.

INSTALLATION AND USAGE
•	To run this project locally, clone the repository and install the necessary libraries:
>git clone https://github.com/codebrew09/Machine-Learning.git
>cd Machine-Learning
>pip install -r requirements.txt

Running the notebook:
1.	Open the notebook Customer_Happiness_Project_Final.ipynb using Jupyter or any compatible notebook editor.
2.	Run the cells sequentially to execute the code and view results.

ACCOMPLISHMENTS
•	Successfully implemented machine learning models for customer happiness prediction.
•	Achieved high accuracy with hyperparameter optimization and feature selection.
•	Created visualizations to aid in understanding model performance and predictions.

FUTURE WORK
•	Exploring More Models: Test other models such as Random Forest, XGBoost, or deep learning models for comparison.
•	Real-Time Application: Implementing the model into a real-time feedback system where predictions can be made based on live customer feedback.
•	Feature Engineering: Additional features such as customer demographics or order history could be incorporated to improve the model’s predictive power.

