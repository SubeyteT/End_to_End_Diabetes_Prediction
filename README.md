# Diabetes_Prediction
This project includes data preprocessing of the dataset, feature engineering and comparative prediction models with different machine learning methods.

**Aim of the project:** 
To develop and compare machine learning models that will predict if one has diabetes(1) or not(0) according to their specifications.

## Dataset Information
Dataset is a part of big dataset kept in National Diabetes-Digestion-Kidney Diseases Institute of USA. The research is done with 768 Pima Indian women who are at least 21 years old and lives in Phoenix in USA. The dataset includes 768 observations and 8 numerical independent variables. 
Target variable: "Outcome". (1) refers to having diabetes ad (0) refers to not having diabates.

##### Variables:
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)2)
DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
Age: Age (years)
Outcome: Class variable (0 if non-diabetic, 1 if diabetic)

You can achieve the public [dataset](https://archive.ics.uci.edu/ml/datasets/diabetes).

## Content
- Data load and 
- Exploratory data analysis (includes graphs)
- Feature engineering
- Outlier detection
- Missing value detection
- Encoding (Label, rare and one-hot encoding)
- Scaling
- Regression Models (Logistic and Linear)

### Summary of feature engineering:
Correlation between variables is reviewed before preprocessing steps. Also, some literature research done and expert (a doctor) opinion is received from for better observations.


- Blood pressure, glucose, skin thickness, insulin and BMI values can't be 0. So they shoul be considered as NA.
- Glucose affects blood pressure.
- From a broader perspective 21 and older female skin thickness range is 7-27 (mm) 
![MicrosoftTeams-image](https://user-images.githubusercontent.com/83431435/124386108-637fbc00-dce1-11eb-8e59-bb67122ae2cc.png)
- Optimal Glucose level range is 70-100. Glucose level < 70: low, glucose level between 100-125 is risky and glucose level > 125 is potential diabetes
- Insulin level is high distributed. But mostly differs between 40-460. So, observations higher than 460 can be considered as outliers.
- Also pregnancy has correlation with outcome, it may trigger diabetes.


## Machine Learning Methods
Until now, logistic regression is applied. Standard scaling is chosen upon some trials. 
Here are test results for logistic regression method:
- ACCURACY: 0.7727272727272727
- PRECISION: 0.7083333333333334
- RECALL: 0.6181818181818182
- F1: 0.6601941747572815
- AUC: 0.8741965105601469


New ML models will be uploaded.







Referances:
https://doi.org/10.2337/diacare.12.5.309










