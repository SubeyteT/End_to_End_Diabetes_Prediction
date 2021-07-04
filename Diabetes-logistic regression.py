import numpy as np
import seaborn as sns
import pandas as pd
from helpers.helpers import *
from helpers.eda import *
from helpers.data_prep import *
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, \
    mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


def load():
    data = pd.read_csv("datasets/diabetes.csv")
    return data

df = load()
df.head()
df.corr()
df.describe().T

df.isnull().sum()

#################### Exploratory data analysis:
cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_summary(df, "Outcome")

for col in num_cols:
    num_summary(df, col, plot=True)

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)



# Filling 0 values with NA values:
na_list = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[na_list] = df[na_list].replace(0,np.NaN)
df.isnull().sum()


#############################################
# 1. Feature Engineering
#############################################


################ Pregnancy History ############### :NO NA
df.loc[(df["Pregnancies"] == 0), "New_Pregnancies"] = "No"
df.loc[(df["Pregnancies"] >= 1), "New_Pregnancies"] = "Yes"
df["New_Pregnancies"].value_counts()


################ Age ############### : NO NA

quartiles = [0,24,29,41, int(df["Age"].max())]
mylabels = ["Young", "Mature","Old-Mature", "Old"]

# Categorize Age
df["New_Age_Cat"] = pd.cut(df["Age"], quartiles, labels=mylabels)
df.head()
df.groupby("New_Age_Cat")["Outcome"].mean()

df.groupby("Outcome")["Age"].count()
df.groupby("Outcome")["Age"].mean()
# Outcome
# 0   31.190
# 1   37.067


################ BMI ###############:

# Outliers:
df.loc[df["BMI"] > 42, "BMI"]  # 74
check_outlier(df, "BMI")
outlier_thresholds(df, "BMI", q1=0.15) # (13.849999999999998, 50.25)
replace_with_thresholds(df, "BMI")

# Filling Missing Values:
df["BMI"] = df["BMI"].fillna(df.groupby("New_Age_Cat")["BMI"].transform("mean"))

df.loc[(df["BMI"] >= 13.5) & (df["BMI"] <= 17.5), "New_BMI_Cat"] = "Thin" # hiç yok
df.loc[(df["BMI"] >= 17.5) & (df["BMI"] <= 24.5), "New_BMI_Cat"] = "Ideal"
df.loc[(df["BMI"] >= 24.5) & (df["BMI"] <= 29.5), "New_BMI_Cat"] = "Extra_Weight"
df.loc[(df["BMI"] >= 29.5), "New_BMI_Cat"] = "Obese"

# Ther might be a meaningful relationship
df.groupby("New_BMI_Cat")["Outcome"].mean()
df.groupby("New_BMI_Cat")["Outcome"].count()

df.corr()

################# SkinThickness ###############:

# Outliers:
outlier_thresholds(df, "SkinThickness")
replace_with_thresholds(df, "SkinThickness")
check_outlier(df, "SkinThickness")

# Filling Missing Values:
df["SkinThickness"] = df["SkinThickness"].fillna(df.groupby(["New_Age_Cat", "New_BMI_Cat"])["SkinThickness"].transform("mean"))

df.loc[(df["SkinThickness"] <= 13), "New_Skin_Cat"] = "Thin"
df.loc[(df["SkinThickness"] > 13) & (df["SkinThickness"] <= 29), "New_Skin_Cat"] = "Healthy"
df.loc[(df["SkinThickness"] > 29), "New_Skin_Cat"] = "AtRisk"

df.groupby("New_Skin_Cat")["Outcome"].mean()
df.groupby("New_Skin_Cat")["Outcome"].count()

#### Feature Investigation  ############### :
df.groupby("New_BMI_Cat").agg({"Outcome":["mean","count"],
                               "Age": ["mean", "count"],
                               "New_Skin_Cat": "count"})

df.groupby(["New_Skin_Cat","New_BMI_Cat"]).agg({"Outcome":["mean","count"],
                                                "Age": ["mean","count"]})

df.groupby(["New_Skin_Cat","New_BMI_Cat","New_Age_Cat"]).agg({"Outcome":["mean","count"]})


############### Glucose ############### :

# Outliers:
check_outlier(df, "Glucose")
outlier_thresholds(df, "Glucose")
replace_with_thresholds(df, "Glucose")

# Filling Missing Values:
df["Glucose"] = df["Glucose"].fillna(df.groupby(["New_Age_Cat","New_Pregnancies", "New_BMI_Cat"])["Glucose"].transform("mean"))
df["Glucose"].isnull().sum()

df.loc[(df["Glucose"] <= 70), "New_Glucose_Cat"] = "Low"
df.loc[(df["Glucose"] >= 70) & (df["Glucose"] <= 100), "New_Glucose_Cat"] = "Normal"
df.loc[(df["Glucose"] >= 100) & (df["Glucose"] <= 125), "New_Glucose_Cat"] = "AtRisk"
df.loc[(df["Glucose"] >= 125), "New_Glucose_Cat"] = "PotentialDiabetes"

df.groupby("New_Glucose_Cat").agg({"Outcome": ["mean", "count"]})

df.groupby(["New_BMI_Cat", "New_Glucose_Cat","New_Age_Cat",]).agg({"Outcome":["mean","count"]})

df.loc[(df["New_BMI_Cat"] == "Ideal") & (df["New_Age_Cat"] == "Young") & (df["Outcome"] == 1)]  # 1 person
df.loc[(df["New_BMI_Cat"] == "Ideal") & (df["New_Age_Cat"] == "Young") & (df["New_Glucose_Cat"] == "Low")]

############### Insulin ############### :

# Filling Missing Values:
df["Insulin"] = df["Insulin"].fillna(df.groupby(["New_Age_Cat","New_Glucose_Cat", "New_BMI_Cat"])["Glucose"].transform("mean"))
df["Insulin"].isnull().sum()

# Outliers:
outlier_thresholds(df,"Insulin", q3=0.95)
replace_with_thresholds(df, "Insulin", q3=0.95)

labels = ["low", "mid", "high"]
bins = [0,40,460,int(df["Insulin"].max())]
df["New_Insulin_Cat"] = pd.cut(df["Insulin"], bins, labels=labels)
df["New_Insulin_Cat"].value_counts()
df.groupby("New_Insulin_Cat")["Outcome"].mean()  # Categories are meaningful

############### Blood Pressure ############### :

# No need for outlier replacement
check_outlier(df, "BloodPressure")
outlier_thresholds(df, "BloodPressure")

# Filling Missing Values:
df["BloodPressure"] = df["BloodPressure"].fillna(df.groupby(["New_Age_Cat", "New_Pregnancies", "New_Glucose_Cat", "New_BMI_Cat"])["BloodPressure"].transform("mean"))
df["BloodPressure"].isnull().sum()  # 1 değer NA kaldı
df["BloodPressure"] = df["BloodPressure"].fillna(df.groupby(["New_Age_Cat"])["BloodPressure"].transform("mean"))
# 0 NA left

df.loc[(df["BloodPressure"] <= 70), "New_BP_Cat"] = "Hipotansiyon"
df.loc[(df["BloodPressure"] >= 70) & (df["BloodPressure"] <= 90), "New_BP_Cat"] = "Optimal"
df.loc[(df["BloodPressure"] >= 90), "New_BP_Cat"] = "Hipertension"

#############################################
# 2. Encoding
#############################################

##### Label Encoding
binary_cols = [col for col in df.columns
               if df[col].dtype == "O"
               and df[col].nunique() == 2]

df[binary_cols].head(5)
for col in binary_cols:
    label_encoder(df, col)

##### Rare Encoding

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "Outcome", cat_cols)
df = rare_encoder(df, 0.01)

#### One_Hot Encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols, drop_first=True)
df.head()
df.shape  # 10 new vars

useless_cols = [col for col in df.columns if df[col].nunique() == 2  and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]  # NONE

#############################################
# 3. Standart Scaler
#############################################

num_cols

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

check_df(df)
df.head()

#############################################
# 4. Logistic Regression
#############################################

############ Model:
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

log_model = LogisticRegression().fit(X_train, y_train)
log_model.intercept_  # -0.99035677
log_model.coef_  #[[ 0.35826797,  0.93470238, -0.13107846, -0.09319875, -0.08377649,
         # 0.36938811,  0.21485785, -0.13378988, -0.53843596,  0.78283739,
         # 1.03205356,  1.31026323, -0.87303999,  0.3860559 , -0.24562649,
         # -0.12571632, -0.55466294, -0.5486778 ,  0.10415693, -0.24175736,
         # 0.4619494 ,  0.02199293,  0.02471568]]

############ Tahmin:
y_pred = log_model.predict(X_train)

# Train Accuracy
y_pred = log_model.predict(X_train)
accuracy_score(y_train, y_pred)

############ Test:
# AUC Score  y_prob
y_prob = log_model.predict_proba(X_test)[:, 1]

# y_pred for other metrics
y_pred = log_model.predict(X_test)

# CONFUSION MATRIX
confusion_matrix(y_test,y_pred)

# ACCURACY
accuracy_score(y_test, y_pred)  # 0.7727272727272727

# PRECISION
precision_score(y_test, y_pred)  #  0.7083333333333334

# RECALL
recall_score(y_test, y_pred)  # 0.6181818181818182

# F1
f1_score(y_test, y_pred)   # 0.6601941747572815

# AUC
roc_auc_score(y_test, y_prob)   # 0.8741965105601469

# Classification report
print(classification_report(y_test, y_pred))