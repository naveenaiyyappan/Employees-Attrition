#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("C:/Users/Ivin/OneDrive/Desktop/general_data.csv")         ### 1


# In[3]:


data


# In[4]:


data.head()


# In[5]:


len(data)


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data.info()


# In[9]:


Attrition_count = data["Attrition"].value_counts()


# In[10]:


Attrition_count


# In[11]:


plt.figure(figsize=(8, 6))
sns.countplot(x='Attrition', data=data)
plt.title('Attrition Distribution')
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.show()


# In[12]:


data_1 =pd.read_csv("C:/Users/Ivin/OneDrive/Desktop/in_time.csv")  ####2


# In[13]:


data_1


# In[14]:


data_1.head()


# In[15]:


data_1.shape


# In[16]:


data_1.columns


# In[17]:


len(data_1)


# In[18]:


data_1.info()


# In[19]:


data_2 = pd.read_csv("C:/Users/Ivin/OneDrive/Desktop/out_time.csv")      ###3


# In[20]:


data_2


# In[21]:


data_2.head()


# In[22]:


data_2.shape


# In[23]:


data_2.columns


# In[24]:


len(data_2)


# In[25]:


data_2.info()


# In[26]:


data_3 = pd.read_csv("C:/Users/Ivin/OneDrive/Desktop/manager_survey_data.csv")    ### 4


# In[27]:


data_3.head()


# In[28]:


data_3.shape


# In[29]:


data_3.columns


# In[30]:


data_3.info()


# In[31]:


len(data_3)


# ###### data preprocessing

# In[32]:


data.isnull().sum()


# In[33]:


data.isnull().sum()/len(data)*100


# In[34]:


data_1.isna().sum()


# In[35]:


data_1.isnull().sum()/len(data_1)*100


# In[36]:


data_2.isnull().sum()


# In[37]:


data_2.isnull().sum()/len(data_2)*100


# In[38]:


data_3.isnull().sum()


# In[39]:


data_3.isna().sum()/len(data_3)*100


# ###### check duplicate values in dataset

# In[40]:


data.duplicated()


# In[41]:


data_1.duplicated()


# In[42]:


data_2.duplicated()


# In[43]:


data.drop_duplicates()


# In[44]:


data_1.drop_duplicates()


# In[45]:


data_2.drop_duplicates()


# In[46]:


data_3.duplicated()


# In[47]:


data_3.drop_duplicates()


# In[48]:


data.corr()


# In[49]:


data.head()


# In[50]:


catergorical_columns=[i for i in data.columns if data[i].dtype=="object"]


# In[51]:


catergorical_columns


# In[52]:


for cat in catergorical_columns:
    plt.pie(data[cat].value_counts(),labels=data[cat].value_counts().values)
    plt.title(cat +"Distribution")
    plt.legend(data[cat].value_counts().index)
    plt.show()


# In[53]:


df=data.drop(["EducationField","Over18","StandardHours"],axis=1)


# In[54]:


num_col=[i for i in data.columns if i not in catergorical_columns]


# In[55]:


num_col


# In[56]:


for numeric in num_col:
    plt.figure(figsize=(14,7))
    sns.countplot(x=numeric,hue="Attrition",data=data)
    plt.title("attrition by"+numeric)
    plt.show()


# In[57]:


data.hist(figsize=(15,15))
plt.tight_layout()
plt.show()


# In[58]:


sns.kdeplot(data.loc[data["Attrition"]=="No","Age"],label="TotalWorkingYears")
sns.kdeplot(data.loc[data["Attrition"]=="Yes","Age"],label="TotalWorkingYears")
plt.legend()
plt.show()                              #### total working years 
                                        #### mostly the 25-30 years pepole can easily leave the job


# In[59]:


sns.kdeplot(data.loc[data["Attrition"]=="NO","MonthlyIncome"],label ="Age")
sns.kdeplot(data.loc[data["Attrition"]=="Yes","MonthlyIncome"],label ="Age")
plt.legend()           #### Age
plt.show()             #### more than 20000-45000 salary person can leave the job


# In[60]:


sns.kdeplot(data.loc[data["Attrition"]=="NO","Age"],label ="JobLevel")
sns.kdeplot(data.loc[data["Attrition"]=="Yes","Age"],label ="JobRole")
plt.legend()              #### JobRole
plt.show()                #### small age employees can there job for there promotion (ie.jobRole)


# In[61]:


data.nunique()


# In[62]:


print(data["JobRole"])


# In[63]:


sns.kdeplot(data.loc[data["Attrition"]=="NO","Age"],label ="JobLevel")
sns.kdeplot(data.loc[data["Attrition"]=="Yes","Age"],label ="JobLevel")
plt.legend()                                    #### JobLLevel
plt.show()                                      #### 40 years old employees can change there job due to there JobLevel


# In[64]:


##### after the Age most of the people leave there job due to salary issues
sns.boxplot(y=data["MonthlyIncome"],x=data["JobRole"])
plt.grid(True,alpha=1)
plt.tight_layout()
plt.show()


# In[65]:


plt.figure(figsize=(15,7))
sns.catplot(x="JobRole",hue="Attrition",data=data,kind="count",legend=False)
plt.title('Count of Attrition in Different Job Roles', fontsize=16)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


# In[66]:


from sklearn.preprocessing import LabelEncoder


# In[67]:


data_label = data.copy()


# In[68]:


data_label.head()


# In[69]:


label_encoder=LabelEncoder()
for cat in catergorical_columns:
    data_label[cat]=label_encoder.fit_transform(data_label[cat])


# In[70]:


data_label.head()


# In[71]:


data_label.info()


# In[72]:


data_label.isnull().sum()/len(data_label)*100


# In[73]:


data_label["NumCompaniesWorked"].mean()


# In[74]:


data_label["NumCompaniesWorked"].median()


# In[75]:


data_label["NumCompaniesWorked"].mode()


# In[76]:


data_label["NumCompaniesWorked"]=data_label["NumCompaniesWorked"].fillna(data_label["NumCompaniesWorked"].median())


# In[77]:


data_label["TotalWorkingYears"].mean()


# In[78]:


data_label["TotalWorkingYears"].median()


# In[79]:


data_label["TotalWorkingYears"].mode()


# In[80]:


data_label["TotalWorkingYears"]=data_label["TotalWorkingYears"].fillna(data_label["TotalWorkingYears"].median())


# ###### Logistic Regression

# In[81]:


from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


# In[82]:


X = data_label.drop(["Attrition"],axis =1)###  Feature
Y = data_label["Attrition"]###b Traget
X.columns


# In[83]:


X.columns


# In[84]:


Y


# ###### Train test split

# In[85]:


from sklearn.model_selection import train_test_split


# In[86]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.3,random_state=107)


# In[87]:


X_train.columns


# ###### model Build

# In[88]:


model_build = LogisticRegression(solver='lbfgs', max_iter=100000) # increase iter to void limited int

model_build = model_build.fit(X_train,Y_train)


# In[89]:


model_build


# In[90]:


X_train.columns


# In[91]:


print('The score of Logistic model on all variables of train set is: {0:0.4f}'.format(model_build.score(X_train, Y_train)))
print('The score of Logistic model on all variables of test set is: {0:0.4f}'.format(model_build.score(X_test, Y_test)))


# ###### confusion matrix

# In[92]:


#### traget predict
Y_test_pred = model_build.predict(X_test)


# In[93]:


Y_test_pred


# In[94]:


from sklearn.metrics import confusion_matrix


# In[95]:


matrix = confusion_matrix(Y_test,Y_test_pred)


# In[96]:


matrix 


# In[97]:


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
disp_12 = ConfusionMatrixDisplay(matrix, display_labels = ['No','Yes'])
disp_12.plot()
plt.title('Confustion Matrix of Model 1 with all variables on test set')
plt.show()


# In[98]:


from sklearn.metrics import classification_report


# In[99]:


print('Classification Report of Model 1 on test set is:')
print(classification_report(Y_test, Y_test_pred))


# ###### Features

# In[100]:


F = ['Age', 'BusinessTravel',  'DistanceFromHome', 'EmployeeID', 'Gender', 'JobRole', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel','TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']


# In[101]:


x = data_label [F]
y = data_label["Attrition"]


# In[102]:


print(x,y)


# In[103]:


X_train_1,X_test_1,Y_train_1,Y_test_1 = train_test_split(x,y,test_size =0.3,random_state = 107)


# In[104]:


X_train_1,X_test_1,Y_train_1,Y_test_1 


# In[105]:


model_build_1 = LogisticRegression(solver='lbfgs', max_iter=100000) # increase iter to void limited int

model_build_1 = model_build_1.fit(X_train_1,Y_train_1)


# In[106]:


model_build_1


# In[107]:


print('The score of 2nd Logistic model of selected variables on train set is: {0:0.4f}'.format(model_build_1.score(X_train_1,Y_train_1)))
print('The score of 2nd Logistic model ofselected variables on test set is: {0:0.4f}'.format(model_build_1.score(X_test_1,Y_test_1)))


# In[108]:


y_pred_1 = model_build_1.predict(X_test_1)


# In[109]:


y_pred_1


# In[110]:


cm_22 = confusion_matrix(Y_test_1,y_pred_1)


# In[111]:


cm_22 


# In[112]:


disp_22 = ConfusionMatrixDisplay(cm_22, display_labels = ['No','Yes']) 
disp_22.plot()
plt.title('Confustion Matrix of Model 2 with selected variables on test set')
plt.show()


# In[113]:


print('Classification Report of Model 2 on test set is:')
print(classification_report(Y_test_1,y_pred_1))


# ###### Dash board creation

# In[114]:


### 1)  Employees Attrition level in the department?
### 2)  Employees Attrition level by their Age?
### 3)  Employees Attrition level by their Gender?
### 4)  Employees Attrition level by their BusineeTravel?
### 5)  Employees Attrition level by their Job Role?
### 6)  Employees Attrition level by employees monthly income?
### 7)  Employees Attrition level by their working experience?
### 8)  Employees Attrition level by employees MaritalStatus?+


# In[115]:


import plotly. express as px


# In[116]:


import dash


# In[117]:


import dash_html_components as html
import dash_core_components as dcc


# In[118]:


data.head()


# In[119]:


data.columns


# In[120]:


data["JobRole"]


# In[121]:


attrition = px.bar(data,x=data["Attrition"],y=data["Department"])
attrition.update_layout(title="Atttrition of employees",title_x=0.5)




attrition = px.scatter(data,x=data["Attrition"],y=data["Age"])
attrition.update_layout(title="employee attrition by age",title_x=0.5)



gender=px.bar(data,x=data["Attrition"],y=data["Gender"])
gender.update_layout(title="employee attrition by Gender",title_x=0.5)


Travel=px.bar(data,x=data["Attrition"],y=data["BusinessTravel"])
Travel.update_layout(title="employee attrition by BusinessTravel",title_x=0.25)


jobroles=px.bar(data,x=data["JobRole"],y=data["Attrition"])
jobroles.update_layout(title="employee attrition by JobRole",title_x=0.25)


income=px.bar(data,x=data["MonthlyIncome"],y=data["Attrition"])
income.update_layout(title="employee attrition by income",title_x=0.25)    ### MaritalStatus


working_years=px.bar(data,x=data["TotalWorkingYears"],y=data["Attrition"])
working_years.update_layout(title="employee attrition by working years",title_x=0.5)


MaritalStatus=px.bar(data,x=data["Attrition"],y=data["MaritalStatus"])
MaritalStatus.update_layout(title="employee attrition by maritalstatus",title_x=0.5)


# In[122]:


my_app = dash.Dash(__name__)
my_app.title = "Employee Attrition"

###  1
attrition = px.bar(data,x=data["Attrition"],y=data["Department"])
attrition.update_layout(title="Atttrition of employees",title_x=0.5)

###  2
age = px.scatter(data,x=data["Attrition"],y=data["Age"])
age.update_layout(title="employee attrition by age",title_x=0.5)

###  3
gender=px.bar(data,x=data["Attrition"],y=data["Gender"])
gender.update_layout(title="employee attrition by Gender",title_x=0.5)

###  4
Travel=px.bar(data,x=data["Attrition"],y=data["BusinessTravel"])
Travel.update_layout(title="employee attrition by BusinessTravel",title_x=0.25)

###  5
jobroles=px.bar(data,x=data["JobRole"],y=data["Attrition"])
jobroles.update_layout(title="employee attrition by JobRole",title_x=0.25)

###  6
income=px.bar(data,x=data["MonthlyIncome"],y=data["Attrition"])
income.update_layout(title="employee attrition by income",title_x=0.25)

###  7 
working_years=px.bar(data,x=data["TotalWorkingYears"],y=data["Attrition"])
working_years.update_layout(title="employee attrition by working years",title_x=0.5)


###  8
MaritalStatus=px.bar(data,x=data["Attrition"],y=data["MaritalStatus"])
MaritalStatus.update_layout(title="employee attrition by maritalstatus",title_x=0.5)


my_app.layout = html.Div([
    html.H1("Employee_Attrition",style = {"text-align":"center","color":"Green"}),
    html.Hr(),
    dcc.Graph(figure = attrition),
    dcc.RadioItems(data["Department"].unique(),data["Department"].unique()[0],id="Radio"),
    dcc.Graph(figure = age),
    dcc.Graph(figure = gender),
    dcc.Graph(figure = Travel),
    dcc.Dropdown(data["BusinessTravel"].unique(),data["BusinessTravel"].unique()[0],id="Drop"),
    dcc.Graph(figure = jobroles),
    dcc.Graph(figure = income),
    dcc.Graph(figure = working_years),
    dcc.Graph(figure = MaritalStatus)
    
])
if __name__ == "__main__":
    my_app.run_server(debug = True, port = 8050)

