# Capston-Project_Tariq
Analyzing risk factors for cervical cancer diagnosis
# <font color="blue" style="font-family:Times New Roman"> Capstone Project: Analyzing risk factors for cervical cancer diagnosis</font> 

## Importance of the project: 
***
+ The red color in the world map showing countries with the highest death rate in women due to cervical cancer.
+ The color (red-green-purple-black-ash) shows from the highest to the lowest death rates in women.
+ African countries have the highest death rate. Below, there is a bar graph showing top 5 countries having highest death rates in women due to cervical cancer.

<div class = "span5 alert alert-success">

![Cervical cancer death rate per 100,000 women worldwide](female.png)

<b>Countries with red (mostly in Africa and South America) and green colors show the severest and next severest death rate per 100,000 female.</b>   

* Source: https://www.worldlifeexpectancy.com/cause-of-death/cervical-cancer/by-country/female


## Introduction
This project aims to investigate/analyze risk factors leading to cervical cancer. Cervical cancer is the fourth-most common death cause in women worldwide. Particularly, in low-income countries in Africa, South America, and Asia, cervical cancer is the most common cause of cancer death in women. Smoking, heterogeneity in partnership, and use of contraceptives are the risk factors for this type of cancer. Diagnosis is done by a biopsy. 

# <font color="blue" style="font-family:Times New Roman"> Capstone Project: Analyzing risk factors for cervical cancer diagnosis</font> 

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

countries = ['Mozambique', 'Malawi', 'Burundi', 'Comoros', 'Tanzania']
death_rate = [49.95, 48.45, 46.35, 44.46, 36.16]

countries.reverse() # reversed for plotting so that top one comes first
death_rate.reverse()

# sns.set_style("whitegrid")
# sns.set_context('poster')
sns.set()
plt.barh(countries, death_rate, color="red", alpha=0.65)
plt.title('Top 5 countries with cervical cancer death in women', fontsize=16)
plt.xlabel('Death rate per 100,000 women', fontsize=14)
plt.show()



## Aim of this project
***

<div class = "span5 alert alert-info">

The focus of this project is to find answers to following research questions:
<font color="FireBrick">
+ Which risk factors, for example, number of partners, age of the first intercourse, number of pregnancies, smoking habit, use of hormonal contraceptives, are most prominent/significant?    
+ What kind of patterns exist among risk factors causing to cervical cancer?
+ Which age group are most vulnerable to this cancer? 
+ How factors, for example, by age group along with smoker or non-smoker, number of years of smoking, and number packs per year of smoking are interrelated? 
+ How factors like number of partners (single or heterogenous), pregnancies or non-pregnancies, use of hormonal contraceptives, intrauterine devices (IUD), information on sexually transmitted diseases (STD), and medical report on cervical intraepithelial neoplasia (CIN) influence the cancer diagnosis?
</font></div>

## The dataset
***

The dataset is from the 'Hospital Universitario de Caracas' in Caracas, Venezuela, collected from the Univerversity of California, Irvine, USA Machine Learning Repository  at https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29.  Informations of 858 patients' respective demography, habits, and historic medical records are in the dataset. Several patients decided not to answer some of the questions because of privacy concerns (missing values).

# from IPython.display import Image
# from IPython.core.display import HTML
# Image(url="OneDrive/Documents/SpringBoard/Capstone_Project_1/Cervical_Cancer_data_code/female.png")

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# image = mpimg.imread("female.png")
# plt.imshow(image)
# plt.show()

## **Read data from a csv file**: Replace **missing values** ('?') by NaN using na_values='?'

# Replace missing values ('?') by Nan
df_data = pd.read_csv('risk_factors_cervical_cancer.csv', na_values='?') 

## **Preliminary data exploration -- shape, columns, isnull**

df_data.shape

pd.set_option('display.max_columns', 36)
df_data.head(3)

df_data[df_data['Biopsy']==1].shape

df_data.info()

### Observations: 
***
+ The dataset has 858 instances with 36 features.
+ Out of 858 patients, only 55 patients. i.e., 6.4% paients are diagnozied with malignant biopsy (biopsy = 1).
+ Two features "<i>STDs: Time since first diagnosis</i>" and "<i>STDs: Time since last diagnosis</i>" have only 71 non-null values. Since more than 90% observations are missing in these two features, we have dropped them from our analysis.

df_data_v1 = df_data.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)

df_data_v1.describe()

**See the pattern of patients' age - boxplot and histogram**

print('Patients\' age: Minimum, 25th, Median, 75th, and Maximum:')
print('-------------------------------------------------------')
df_data_v1.Age.describe()

plt.subplot(2,1,1)
circle_red_flier = dict(marker='o', markerfacecolor='r', markersize=8,
                  linestyle='none', markeredgecolor='b')
# cap = dict(linestyle='solid', color='r')
df_data_v1.Age.plot(kind='box', figsize=(12,4), vert=False, flierprops=circle_red_flier, fontsize=14) #, capprops=cap)
plt.xlabel('Patients\' age in years', fontsize=14)
plt.show()

plt.subplot(2,1,2)
df_data_v1.Age.plot(kind='hist', bins=20, figsize=(12,10))
# sns.distplot(a=np.array(df_data_v1.Age), hist=True, norm_hist=True)
plt.xlabel('Patients\' age in years', fontsize=14)
plt.show()

### Key takeways:
***
+ **The distribution of patients' "Age" is positively skewed (left modal).**
+ **Mean = 26.82 years, Min. = 13 years, Max. = 84 years, and Standard deviation = 8.50 years.** 
+ **The boxplot also shows that there are some outliers above the age of 50 years.**

### **Divide the dataset into three parts: patients' (a) behavior, (b) smoking habits, and (c) medical diagnosis**
***
+ Age
+ Number od sexual partners
+ Age at the first intercourse
+ Number of pregnancies
+ Biopsy results: 0 (benign) or 1 (malignant)

df_behavior = df_data_v1[['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Biopsy']]

df_behavior.head(3)

Fill forward NaN values

df_behavior_v1 = df_behavior.fillna(method='ffill')
# df_behavior_v1.isnull().sum()

df_behavior_v1['Biopsy'] = df_behavior_v1['Biopsy'].map({0:'Benign', 1:'Malignant'})

df_behavior_v1.head(3)

sns.set(style='ticks')
sns.pairplot(df_behavior_v1, hue='Biopsy', height=3, markers=[".","s"], aspect=1, kind='scatter') # hue_order=['green', 'red'])
# plt.suptitle("Pairplot showing reltionship among Age, Number of partners, Age of first intercourse, Number of pregnancies", fontsize=14)
# plt.grid(True)
plt.show()

fig, ax_1 = plt.subplots(figsize=(15,7))
sns.boxplot(x="Age", y="Num of pregnancies", hue="Biopsy", data=df_behavior_v1, orient="v", ax=ax_1)
plt.grid(True)
plt.show()

fig, ax_1 = plt.subplots(figsize=(15,7))
sns.boxplot(x="Age", y="Number of sexual partners", hue="Biopsy", data=df_behavior_v1, orient="v", ax=ax_1)
plt.grid(True)
plt.show()

### **Smoking habits**

df_smoking = df_data_v1[['Age', 'Smokes (years)', 'Smokes (packs/year)', 'Biopsy']].fillna(method='ffill')
df_smoking.isnull().sum()

# df_smoking.head(3)
df_smoking_lg = df_smoking.copy()

df_smoking_lg.shape, df_smoking_lg.head(3)

df_smoking['Biopsy'] = df_smoking['Biopsy'].map({0:'Benign', 1:'Malignant'})

sns.pairplot(df_smoking, hue='Biopsy', markers=['.','s'], height=3, aspect=1, kind='scatter') #, diag_kind='kde')
plt.show()

fig, ax_1 = plt.subplots(figsize=(15,8))
sns.boxplot(x="Age", y="Smokes (years)", hue="Biopsy", data=df_smoking, orient="v", ax=ax_1)
plt.grid(True)
plt.show()

fig, ax_1 = plt.subplots(figsize=(15,10))
sns.boxplot(x="Age", y="Smokes (packs/year)", hue="Biopsy", data=df_smoking, orient="v", ax=ax_1)
plt.grid(True)
plt.show()

df_hormone_IUD = df_data_v1[['Age', 'Hormonal Contraceptives (years)', 'IUD (years)', 'Biopsy']]

df_hormone_IUD.head(3)

df_hormone_IUD['Biopsy'] = df_hormone_IUD['Biopsy'].map({0:'Benign', 1:'Malignant'})

df_hormone_IUD.head(3)

_ = sns.pairplot(df_hormone_IUD, hue='Biopsy', markers=['.','s'], height = 3, aspect=1, kind='scatter', diag_kind='kde')
_ = plt.show()

# df_data_v1["Biopsy"] = df_data_v1["Biopsy"].map({0:"Benign", 1:"Malignant"})

## Droping outliers 

# df_data_v2 = df_data_v1[df_data_v1.Age >=18] 
# df_data_v2 = df_data_v2[df_data_v2.Age <= 40]

# df_data_v1.isnull().sum()

df_data_v2 = df_data_v1[(df_data_v1.Age >= 18) & (df_data_v1.Age <= 40)]

df_data_v2.shape

df_data_v2["Biopsy"] = df_data_v2["Biopsy"].map({0:"Benign", 1:"Malignant"})

df_data_v3 = df_data_v2.fillna(method='ffill', inplace=True) # fillna geneartes a new object
# df_data_v2.isnull().sum()

type(df_data_v3)

fig, ax1 = plt.subplots(figsize=(13,5))
sns.boxplot(x="Age", y="Smokes (years)", hue="Biopsy", data = df_data_v2, ax = ax1)
ax1.tick_params(labelsize=15)
ax1.set_xlabel("Age group between 18 and 40", fontsize=16)
ax1.set_title("Pattern of smoking years by patients' age and biopsy", fontsize=16)
ax1.grid(True)

df_smoking_v1 = df_data_v2[["Age", "Smokes (years)", "Smokes (packs/year)", "Biopsy"]]

sns.pairplot(df_smoking_v1, hue="Biopsy", markers=['.', 's'], height = 3, kind = 'scatter')
plt.show()

fig, ax1 = plt.subplots(figsize=(13,5))
sns.boxplot(x="Age", y="Number of sexual partners", hue="Biopsy", data = df_data_v2, ax = ax1)
ax1.tick_params(labelsize=15)
ax1.set_xlabel("Age group between 18 and 40", fontsize=16)
ax1.set_title("Partners by patients' age and biopsy", fontsize=16)
ax1.grid(True)

fig, ax1 = plt.subplots(figsize=(13,5))
sns.boxplot(x="Age", y="Smokes (packs/year)", hue="Biopsy", data = df_data_v2, ax = ax1)
ax1.tick_params(labelsize=15)
ax1.set_xlabel("Age", fontsize=16)
ax1.set_title("Pattern of smoking packs/year by patients' age and biopsy", fontsize=16)
ax1.grid(True)

## Statistical Inference: Hypothesis testing 

Let $\mu_m$ be the mean smoking years by **malignant** pateients 
and $\mu_b$ be the mean smoking years by **benign** patients

$H_0$: $\mu_m - \mu_b = 0$   

$H_1$: $\mu_m - \mu_b \neq 0$

# df_smoking_v1 
# print(df_smoking_v1.isnull().sum())
# df_smoking_v1.fillna(method='ffill', axis=1)
# print(df_smoking_v1.isnull().sum())

mean_smoke_years_malignant = np.mean(df_smoking_v1[df_smoking_v1['Biopsy']=='Malignant']['Smokes (years)'])
mean_smoke_years_benign = np.mean(df_smoking_v1[df_smoking_v1['Biopsy']=='Benign']['Smokes (years)'])

print("Mean smoking years by malignant patients: {:.2f}".format(mean_smoke_years_malignant))
print("Mean smoking years by benign patients:    {:.2f}".format(mean_smoke_years_benign))

smoking_years_malignant = df_smoking_v1[df_smoking_v1['Biopsy']=='Malignant']['Smokes (years)']
smoking_years_benign = df_smoking_v1[df_smoking_v1['Biopsy']=='Benign']['Smokes (years)']

mean_smoke = np.mean(np.concatenate((smoking_years_malignant, smoking_years_benign)))

shifted_mean_malignant = smoking_years_malignant - np.mean(smoking_years_malignant) + mean_smoke
shifted_mean_benign    = smoking_years_benign - np.mean(smoking_years_benign) + mean_smoke


# print(mean_smoke)

def mean_diff(data_malignant, data_benign):
    mean_malignant = np.mean(data_malignant)
    mean_benign    = np.mean(data_benign)
    return mean_malignant - mean_benign

sample_mean_diff = mean_diff(smoking_years_malignant, smoking_years_benign)
print("Sample mean diff: {:.2f}".format(sample_mean_diff))

def draw_replicates(data, size=1):
    bs_rep_sample = np.zeros(size)
    for i in range(size):
        bs_rep_sample[i] = np.mean(np.random.choice(data, size=len(data)))
    return bs_rep_sample        

np.random.seed(42)
smoking_rep_malignant = draw_replicates(shifted_mean_malignant, size=10000)
smoking_rep_benign = draw_replicates(shifted_mean_benign, size=10000)
# len(smoking_rep_malignant)
mean_rep = smoking_rep_malignant - smoking_rep_benign
print()

_ = sns.distplot(mean_rep, bins=50)
plt.axvline(sample_mean_diff, color='r', linestyle='-', linewidth=2)
plt.grid(True)

confidence_interval_95 = np.percentile(mean_rep, [2.5, 97.5])
print("95% confidence interval of mean difference between smoking years by malignant and benign patients:\n ", confidence_interval_95)

p_value = np.sum(mean_rep >= sample_mean_diff)/len(mean_rep)
print("p_value: ", p_value)

## Classification using Scikit learn Logistic Regression

We will apply logistic regression to classfy malignant and benign patients by their age and smoking habits (smoking years and number of packs)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import scale

import warnings
warnings.filterwarnings('ignore')

df_data_lg = df_data_v1.fillna(method='ffill')

# X = df_data_lg[smoking_features].values
# X_scaled = scale(X)

def select_features_and_target(df, predictors, target):
    X = df[predictors]
    y = df[target]
    return X, y

def model(X, y, testsize = 0.2, randomstate=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= testsize, random_state = randomstate)
    log_reg = LogisticRegression(class_weight='balanced')
    
    penalties = ['l1', 'l2']
    Cs = np.logspace(-3,1,10)
    hyper_params = dict(C = Cs, penalty = penalties)

    # Develop GridSearchCV
    gs_cv = GridSearchCV(estimator=log_reg, param_grid = hyper_params, cv=5)

    # Conduct Grid Search
    best_model = gs_cv.fit(X_train, y_train)
    
    print("Best penalty (from training data):", best_model.best_estimator_.get_params()['penalty'])
    print("Best C (from training data):      ", best_model.best_estimator_.get_params()['C'])
    print("Best score (from training data):  ", best_model.best_score_)

    
    y_pred = gs_cv.predict(X_test)
    y_pred_prob = gs_cv.predict_proba(X_test)[:,1]

    fpr, tpr, thresolds = roc_curve(y_test, y_pred_prob)
    roc_score = roc_auc_score(y_test, y_pred_prob)
    
    print("\n======================")
    print("Classification Report")
    print("======================")
    print(classification_report(y_test, y_pred))
    
    # plot ROC Curve
    print("ROC SCORE: {:.2f}".format(roc_score))
    plt.plot([0,1],[0,1], 'k--')
    plt.plot(fpr, tpr, label="Logistic Regression")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Logistic Regression ROC Curve")
    

smoking_features = ['Age', 'Smokes (years)', 'Smokes (packs/year)']
y_feature = ['Biopsy']

X, y = select_features_and_target(df_data_lg, smoking_features, y_feature)
model(X, y, testsize=0.2)

df_data_lg_age_18_40 = df_data_lg[(df_data_lg.Age >= 18) & (df_data_lg.Age <=40)]

print("REGRESSION RESULTS (DROPPING OUTLIERS)")
print("======================================\n")
X, y = select_features_and_target(df_data_lg_age_18_40, smoking_features, y_feature)
model(X,y, testsize=0.2)

# df_data_lg.columns
behavior_features = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies']

X, y = select_features_and_target(df_data_lg_age_18_40, behavior_features, y_feature)
model(X, y)



X = df_data_lg_age_18_40.drop(['Biopsy'], axis=1).values

model(X, y)
