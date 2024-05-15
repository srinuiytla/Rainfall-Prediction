#!/usr/bin/env python
# coding: utf-8

# # RainFall Prediction
# 
# ## Problem Statement
# 
# #### Build an efficient Classification Model that should predict whether it Rains Tomorrow or not, using the dataset
# 
# ## Features description
# 
# #### Date---The date of observation
# 
# #### Location---The common name of the location of the weather station
# 
# #### MinTemp---The minimum temperature in degrees celsius
# 
# #### MaxTemp---The maximum temperature in degrees celsius
# 
# #### Rainfall---The amount of rainfall recorded for the day in mm
# 
# #### Evaporation---The so-called Class A pan evaporation (mm) in the 24 hours to 9am
# 
# #### Sunshine---The number of hours of bright sunshine in the day.
# 
# #### WindGustDir---The direction of the strongest wind gust in the 24 hours to midnight
# 
# #### WindGustSpeed---The speed (km/h) of the strongest wind gust in the 24 hours to midnight
# 
# #### WindDir9am---Direction of the wind at 9am
# 
# #### WindDir3pm---Direction of the wind at 3pm
# 
# #### WindSpeed9am---Wind speed (km/hr) averaged over 10 minutes prior to 9am
# 
# #### WindSpeed3pm---Wind speed (km/hr) averaged over 10 minutes prior to 3pm
# 
# #### Humidity9am---Humidity (percent) at 9am
# 
# #### Humidity3pm---Humidity (percent) at 3pm
# 
# #### Pressure9am---Atmospheric pressure (hpa) reduced to mean sea level at 9am
# 
# #### Pressure3pm---Atmospheric pressure (hpa) reduced to mean sea level at 3pm
# 
# #### Cloud9am---Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of                    eigths. It records how many eigths of the sky are obscured by cloud. A 0 measure indicates                    completely clear sky whilst an 8 indicates that it is completely overcast.
# 
# #### Cloud3pm---Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description                of the values
# 
# #### Temp9am---Temperature (degrees C) at 9am
# #### RISK_MM---The amount of next day rain in mm. Used to create response variable RainTomorrow. A kind of  measure of the "risk".
# 
# #### RainTomorrow---The target variable. Did it rain tomorrow?
# 
# #### Temp3pm---Temperature (degrees C) at 3pm
# 
# #### RainToday---Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
# 

# In[76]:


# Importing required libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[77]:


# Importing the data

df = pd.read_csv(r"C:\Users\shinu\Downloads\Rainfall Prediction\weatherAUS.csv")


# In[78]:


df


# In[79]:


df.head()


# In[81]:


df.tail()


# ## EDA

# In[170]:


df['WindGustDir'].value_counts().plot(kind="pie",autopct="%.2f%%",figsize=(10,7))
plt.show()


# In[168]:


df['WindDir9am'].value_counts().plot(kind="pie",autopct="%.2f%%",figsize=(10,7))
plt.show()


# In[166]:


df['WindDir3pm'].value_counts().plot(kind="pie",autopct="%.2f%%",figsize=(10,7))
plt.show()


# In[174]:


df['Humidity3pm'].value_counts().plot(kind="bar",figsize=(20,15))
plt.show()


# In[175]:


df['RainTomorrow'].value_counts().plot(kind="pie")
plt.show()


# In[89]:


ax=pd.crosstab(df['WindGustDir'],df['WindDir9am']).plot(kind="bar",stacked=True,figsize = (25,15))
for i in ax.containers:
    ax.bar_label(i)


# In[90]:


ax=pd.crosstab(df['WindDir9am'],df['WindDir3pm']).plot(kind="bar",stacked=True,figsize = (25,15))
for i in ax.containers:
    ax.bar_label(i)


# In[103]:


ax=pd.crosstab(df['RainToday'],df['RainTomorrow']).plot(kind="bar",stacked=True)
for i in ax.containers:
    ax.bar_label(i)


# In[93]:


# Pairplot for whole data tells you the distribution for single column and that single column with all other different columns

sns.pairplot(df,diag_kind='kde')


# In[94]:


# Finding the distribution for each single columnb

df.hist(figsize=(20,15))
plt.show()


# In[35]:


# Checking Correlation between two variabless using Heatmap

k=df.select_dtypes(include=['int','float'])
cor=k.corr()
plt.figure(figsize=(20,15))
sns.heatmap(cor,annot=True)
plt.show()


# In[104]:


# sorting the data based on date (Time based splitting)

df = df.sort_values(by='Date')


# In[105]:


#After sorting the data index's got changed to get that in order we doing reset index

df.reset_index(inplace=True)
df.drop("index",axis = 1,inplace=True)


# In[106]:


#Removing unwanted features, RISK_MM is same as target label hence removing with date and loaction

df.drop(['Date', 'Location','RISK_MM'], axis=1,inplace = True)


# In[108]:


df.head()


# In[110]:


df.dtypes


# ## Chi Square Test

# In[111]:


# Both variables categorical - chi square test of independence

pd.crosstab(df['WindGustDir'],df['WindDir9am'])


# In[112]:


# Null-there is no association between both variables
# Alt-There is association between both variables


# In[113]:


from scipy.stats import chi2_contingency
chi2_contingency(pd.crosstab(df['WindGustDir'],df['WindDir9am']))

#Since Pvalue = 0.0 Not Reject Null


# In[115]:


pd.crosstab(df['WindDir9am'],df['WindDir3pm'])


# In[116]:


chi2_contingency(pd.crosstab(df['WindDir9am'],df['WindDir3pm']))

# Since pvalue=0.0 , Not Reject Null


# ## Data Preprocessing

# In[117]:


#Checking for null valuess

df.isna().sum()


# In[120]:


df.columns


# In[121]:


#Creating a list to fill the Median for numerical columns

Numerical_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
       'Sunshine', 'WindGustSpeed','WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
       'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
       'Temp3pm']


# In[122]:


#Filling the values and making permanent change 

for c in Numerical_cols:
    p = df[c].median()
    df[c] = df[c].fillna(p)


# In[123]:


##Creating a list to fill the Mean for categorical columns


Categorical_cols = ["WindGustDir","WindDir9am","WindDir3pm","RainToday"]


# In[124]:


#Filling the values and making permanent change 


for x in Categorical_cols:
    q = df[x].mode()[0]
    df[x] = df[x].fillna(q)


# In[125]:


df.dtypes


# ## Encoding

# In[126]:


import warnings
warnings.filterwarnings("ignore")


# In[127]:


#After EDA we having some categorical columns i.e., WindGustDir,WindDir9am,WindDir3pm,RainToday,RainTomorrow 


#For Rain Tomorrow column we use LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["RainTomorrow"] = le.fit_transform(df[["RainTomorrow"]])


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["WindGustDir"] = le.fit_transform(df[["WindGustDir"]])


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["WindDir9am"] = le.fit_transform(df[["WindDir9am"]])



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["WindDir3pm"] = le.fit_transform(df[["WindDir3pm"]])


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["RainToday"] = le.fit_transform(df[["RainToday"]])


# In[128]:


df


# ## Splitting the data

# In[129]:


X = df.drop("RainTomorrow",axis = 1)
Y = df["RainTomorrow"]


# In[130]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)


# ## Model Building

# In[131]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier  


# ## Logistic Regression

# In[132]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X,Y)
lr.score(X,Y)


# In[133]:


from sklearn.model_selection import cross_val_score
cv=cross_val_score(lr,X,Y)
print(cv)
print()
print(cv.mean())


# ### Model Evaluation

# In[135]:


RocCurveDisplay.from_predictions(Y_test,lr_pred)
plt.show()


# In[136]:


from sklearn.metrics import confusion_matrix,classification_report,f1_score,RocCurveDisplay
lr_pred = lr.predict(X_test)


# In[137]:


print(confusion_matrix(Y_test,lr_pred))


# In[138]:


print(classification_report(Y_test,lr_pred))


# In[ ]:





# ## DecisionTree Classifier

# In[141]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=10)
dt.fit(X,Y)
dt.score(X,Y)


# In[142]:


cv1=cross_val_score(dt,X,Y)
print(cv1)
print()
print(cv1.mean())


# ### Model Evaluation

# In[143]:


RocCurveDisplay.from_predictions(Y_test,dt_pred)
plt.show()


# In[144]:


from sklearn.metrics import confusion_matrix,classification_report
dt_pred = dt.predict(X_test)


# In[145]:


print(confusion_matrix(Y_test,dt_pred))


# In[146]:


print(classification_report(Y_test,dt_pred))


# In[ ]:





# ## Random forest Classifier

# In[147]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=10)
rfc.fit(X,Y)
rfc.score(X,Y)


# In[148]:


cv2=cross_val_score(rfc,X,Y)
print(cv2)
print()
print(cv2.mean())


# ### Model Evaluation

# In[149]:


from sklearn.metrics import confusion_matrix,classification_report
rfc_pred = rfc.predict(X_test)


# In[150]:


RocCurveDisplay.from_predictions(Y_test,rfc_pred)
plt.show()


# In[151]:


print(confusion_matrix(Y_test,rfc_pred))


# In[152]:


print(classification_report(Y_test,rfc_pred))


# In[ ]:





# ## GradientBoosting Classifier

# In[153]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X,Y)
gbc.score(X,Y)


# In[154]:


cv3=cross_val_score(gbc,X,Y)
print(cv3)
print()
print(cv3.mean())


# ### Model Evaluation

# In[159]:


from sklearn.metrics import confusion_matrix,classification_report
gbc_pred = gbc.predict(X_test)


# In[160]:


RocCurveDisplay.from_predictions(Y_test,gbc_pred)
plt.show()


# In[161]:


print(confusion_matrix(Y_test,gbc_pred))


# In[162]:


print(classification_report(Y_test,gbc_pred))


# In[ ]:





# ## XGBClassifier

# In[163]:


from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X,Y)
xg.score(X,Y)


# ### Model Evaluation

# In[164]:


xg_pred = xg.predict(X_test)


# In[165]:


RocCurveDisplay.from_predictions(Y_test,xg_pred)
plt.show()


# ## Results and Conclusion:
#         
# #### Best Models in terms of accuracy (In my Experiment):
# 
# #### 1) XG Boost Model
# #### 4) Decision Tree Classifier
# #### 5) Gradient Boost Classifier
# 
# ### Best Models in terms of Computation Time (In my Experiment):
# 
# #### 1) Logistic Regression
# #### 2) Decisoin Tree Classifier
# #### 3) XG Boost Model
# 
# 
# ## Conclusion:
# 
# ### The Ensemble model XG Boost is performing well when compared with Other models like Logistic Regression,Random Forest,Decision Tree and GradientBoost. But XG Boost model consumes more time to train the model.
# 

# In[ ]:




