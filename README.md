# Medical_Insurance_Cost_Prediction

Medical-Insurance-Cost-Estimation-Using-Machine-Learning:

This repository contains code for predicting medical charges using linear regression and conducting sensitivity analysis on the model's performance.

Introduction:

Healthcare costs are a significant concern for individuals and families worldwide. Understanding the factors that influence medical charges can help healthcare providers and policymakers better allocate resources and plan interventions. In this project, we use linear regression to predict medical charges based on demographic and health-related factors.

Dataset:
The dataset comprises several features:-

age: Age of the individual.
sex: Gender (male/female).
bmi: Body Mass Index.
children: Number of children/dependents.
smoker: Smoking status (yes/no).
region: Geographical region.
charges: Individual medical costs billed by health insurance.

Tasks:

* Data Preprocessing
* Handle missing or anomalous data.
* Convert categorical variables (sex, smoker, region) into numerical formats using encoding techniques.
* Normalize/standardize numerical features if required.
* Exploratory Data Analysis (EDA)
* Analyze the distribution of key variables (e.g., age, BMI, charges).
* Investigate relationships between features and the target variable (charges).
* Identify potential outliers or influential points.

Model Development:

Split the dataset into training and testing sets.
Implement a linear regression model.
Evaluate model performance using appropriate metrics (e.g., R-squared, Mean Squared Error).
Advanced Analysis
Implement regularized linear models (Ridge, Lasso) to see if they yield better results.
Conduct a sensitivity analysis to understand the robustness of the model.

Files :

Medical_Insurance_Prediction.ipynb: Jupyter Notebook containing code for data preprocessing, EDA, model development, and sensitivity analysis.
insurance.csv: Dataset containing medical charges and related features.
Requirements
Python 3.x
Jupyter Notebook
pandas
scikit-learn
matplotlib

Usage

Clone the repository:

git clone https://github.com/AnanyaB-262005/Medical_Insurance_Prediction.git
Navigate to the project directory:

cd Medical_Insurance_Prediction
Open and run the Jupyter Notebook  Medical_Insurance_Prediction.ipynb to execute the code and analyze the results.

Results :

The linear regression model achieved an R-squared value of 0.78 and a Mean Squared Error of 0.23.
Sensitivity analysis revealed high fluctuation in model performance metrics across different train-test splits, indicating sensitivity to variations in the training data.

Conclusion :

Linear regression provides a baseline model for predicting medical charges based on demographic and health-related factors. Further analysis, including regularized linear models and sensitivity analysis, can help improve model robustness and generalization.

Acknowledgements :

The dataset used in this analysis is sourced from Kaggle.
