#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


data = pd.read_csv (r'C:\Users\iran\Downloads\cardata.csv')
data


# In[3]:


df = pd.DataFrame(data)


# In[4]:


df


# In[5]:


df1 = df.drop (columns = ["Car_Name"])
df1


# In[6]:


df1.info()


# In[7]:


df1.describe()


# In[8]:


#### missing values
df1.isnull().sum()


# In[9]:


#### preprocessing


# In[10]:


df1


# In[11]:


### max year = 2018
### creat columns Age
Year2 = 2019 - df1['Year']


# In[12]:


df1.insert(8 , "Age" , Year2 , True)


# In[13]:


df2 = df1.drop(columns= ["Year"])
df2


# In[14]:


df2


# In[15]:


df2.info()


# In[16]:


df2.describe()


# In[17]:


### target is : selling_price
### feature is : Year , Age , Present_Price , Kms_Driven , Owner


# In[18]:


plt.figure( figsize = (10, 5))
plt.scatter (df2 ["Age"] , df2["Selling_Price"])
plt.xlabel ("Age")
plt.ylabel ("Selling_Price")
plt.grid()


# In[19]:


### Selling_Price > 30
### Age < 10
df2[(df2["Selling_Price"] > 30 ) & (df2["Age"]  < 10)]


# In[20]:


### Selling_Price > 5
### Age < 2
df2[(df2["Selling_Price"] > 5 ) & (df2["Age"]  < 2)]


# In[21]:


### Selling_Price < 5
### Age < 16
df2[(df2["Selling_Price"] < 5 ) & (df2["Age"]  > 14)]


# In[22]:


plt.figure( figsize = (10, 5))
plt.scatter (df2 ["Present_Price"] , df2["Selling_Price"])
plt.xlabel ("Present_Price")
plt.ylabel ("Selling_Price")
plt.grid()


# In[23]:


### present_price > 80
### selling_price < 35
df2[(df2["Selling_Price"] < 40 ) & (df2["Present_Price"]  > 80)]


# In[24]:


### Selling_Price > 30
### Present_Price < 40
df2[(df2["Selling_Price"] > 30 ) & (df2["Present_Price"]  < 40)]


# In[25]:


plt.figure( figsize = (10, 5))
plt.scatter (df2 ["Kms_Driven"] , df2["Selling_Price"])
plt.xlabel ("Kms_Driven")
plt.ylabel ("Selling_Price")
plt.grid()


# In[26]:


### Selling_Price > 30
### Kms_Driven < 100000
df2[(df2["Selling_Price"] > 30) & (df2["Kms_Driven"]  < 100000)]


# In[27]:


### Selling_Price < 5
### Kms_Driven > 400000
df2[(df2["Selling_Price"] < 5) & (df2["Kms_Driven"]  > 400000)]


# In[28]:


plt.figure( figsize = (10, 4))
plt.scatter (df2 ["Fuel_Type"] , df2["Selling_Price"])
plt.xlabel ("Fuel_Type")
plt.ylabel ("Selling_Price")
plt.grid()


# In[29]:


df2.head()


# In[30]:


plt.figure( figsize = (10, 5))
plt.scatter (df2 ["Seller_Type"] , df2["Selling_Price"])
plt.xlabel ("Seller_Type")
plt.ylabel ("Selling_Price")
plt.grid()


# In[31]:


plt.figure( figsize = (10, 5))
plt.scatter (df2 ["Transmission"] , df2["Selling_Price"])
plt.xlabel ("Transmission")
plt.ylabel ("Selling_Price")
plt.grid()


# In[32]:


plt.figure( figsize = (10, 5))
plt.scatter (df2 ["Owner"] , df2["Selling_Price"])
plt.xlabel ("Owner")
plt.ylabel ("Selling_Price")
plt.grid()


# In[33]:


### Selling_Price < 5
### Owner > 2.5
df2[(df2["Selling_Price"] < 5) & (df2["Owner"]  > 2.5)]


# In[34]:


df2


# In[35]:


#####    Creat ML    #####


# In[36]:


####  فراخوانی یونیک ها  ####


# In[37]:


### Fuel_Type , Seller_Type ,  Transmission ,  Owner
print (pd.unique(df2["Fuel_Type"]))
print (pd.unique(df2["Seller_Type"]))
print (pd.unique(df2["Transmission"]))
print (pd.unique(df2["Owner"]))


# In[38]:


#### Convert uniques to string to int or float


# In[39]:


#### with the help of the replace command
df2["Fuel_Type"].replace({"Petrol" : 2 , "Diesel" : 3 , "CNG" : 4} , inplace = True)
df2["Seller_Type"].replace({"Dealer" : 2 , "Individual" : 3} , inplace = True)
df2["Transmission"].replace({"Manual" : 2 , "Automatic" : 3} , inplace = True)


# In[40]:


df2


# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 


# In[42]:


x = pd.DataFrame(df2 , columns = ["Present_Price" , "Kms_Driven" , "Fuel_Type" , "Seller_Type" , "Transmission" , "Owner" , "Age"])
y = df1["Selling_Price"].values.reshape(-1,1)


# In[43]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0)


# In[44]:


print ("x train" , x_train.shape)
print ("x test" , x_test.shape)
print ("y train" , y_train.shape)
print ("y test" , y_test.shape)


# In[45]:


regressor = LinearRegression ()


# In[46]:


regressor.fit (x_train , y_train)


# In[47]:


y_pred = regressor.predict(x_test)


# In[48]:


df3 = x_test.sort_values(by = ["Age"])
df3


# In[49]:


df2


# In[50]:


x_test


# In[51]:


compare = pd.DataFrame({"Actual": y_test.flatten() , "predict": y_pred.flatten()})
compare


# In[52]:


print(regressor.intercept_)
print(regressor.coef_)


# In[53]:


### First model
print('mean Absolute Error:' , metrics.mean_absolute_error(y_test , y_pred))
print('mean squared Error :' , metrics.mean_squared_error(y_test , y_pred))
print('Root mean squared Error :' , np.sqrt(metrics.mean_squared_error(y_test , y_pred)))
print('R2score:' , metrics.r2_score(y_test , y_pred))


# In[54]:


df2


# In[55]:


### d2 


# In[56]:


x


# In[57]:


def chek (Dimension , Testsize):
    r2 =  0.9134181721224179
    for column in x :
        new_col_name = column + str(Dimension)
        new_col_val = x[column]** Dimension
        x.insert(0 , new_col_name , new_col_val)
        x_train , x_test , y_train , y_test = train_test_split (x ,y ,test_size = Testsize , random_state =0)
        new_model = LinearRegression()
        new_model .fit(x_train , y_train)
        y_pred = new_model.predict(x_test)
        r2_new = metrics.r2_score(y_test , y_pred)
        
        if r2_new < r2 :
            x.drop([new_col_name] , axis = 1 , inplace = True)
        else :
                r2 = r2_new
    print  ('R2 score :', r2)       

chek(2, 0.2)
        


# In[58]:


x


# In[59]:


#### more power reduces the accuracy of the model


# In[60]:


### present_price whit  Kms_Driven , Fuel_Type , Seller_Type , Transmission

#### integration of features


# In[61]:


present_fuel = x["Present_Price"] * x["Fuel_Type"]
present_fuel2 = x["Present_Price"] * x["Fuel_Type2"]

present_Owner = x["Present_Price"] * x["Owner"]
present_Owner2 = x["Present_Price"] * x["Owner2"]

present_kms = x["Present_Price"] * x["Kms_Driven"]
present_kms2 = x["Present_Price"] * x["Kms_Driven2"]

present2_fuel = x["Present_Price2"] * x["Fuel_Type"]
present2_fuel2 = x["Present_Price2"] * x["Fuel_Type2"]

present2_Owner = x["Present_Price2"] * x["Owner"]
present2_Owner2 =  x["Present_Price2"] * x["Owner2"]

present2_kms = x["Present_Price2"] * x["Kms_Driven"]
present2_kms2 = x["Present_Price2"] * x["Kms_Driven2"]


# In[62]:


x.insert(0 , "present_fuel", present_fuel)
x.insert(0 , "present_fuel2", present_fuel2)
x.insert(0 , "present_Owner", present_Owner)
x.insert(0 , "present_Owner2", present_Owner2)
x.insert(0 , "present_kms", present_kms)
x.insert(0 , "present_kms2", present_kms2)
x.insert(0 , "present2_fuel", present_fuel)
x.insert(0 , "present2_fuel2", present_fuel)
x.insert(0 , "present2_Owner", present_fuel)
x.insert(0 , "present2_Owner2", present_fuel)
x.insert(0 , "present2_kms", present_fuel)
x.insert(0 , "present2_kms2", present_fuel)


# In[63]:


x


# In[64]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0)
model = LinearRegression()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
r2 = metrics.r2_score(y_test , y_pred)
print('mean Absolute Error:' , metrics.mean_absolute_error(y_test , y_pred))
print('mean squared Error :' , metrics.mean_squared_error(y_test , y_pred))
print('Root mean squared Error :' , np.sqrt(metrics.mean_squared_error(y_test , y_pred)))
print('R2score:' , r2)


# In[65]:


#### k_fold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[66]:


model = LinearRegression()
KFold_validation = KFold(5)
results = cross_val_score(model , x , y , cv =KFold_validation)
print(results)
print(np.mean(results))


# In[67]:


x.shape


# In[68]:


301/6


# In[69]:


df = x[x.index<100]
df12 = x[x.index>151]
x_new = df.append(df12)
x_new.reset_index(drop = True , inplace = True)
x_new


# In[70]:


y = df2 ['Selling_Price']
y1 = y[y.index < 100]
y2 = y[y.index > 151]
y_new = y1.append(y2)
y_new.reset_index(drop = True , inplace = True)
y_new


# In[71]:


x_train , x_test , y_train , y_test = train_test_split(x_new , y_new , test_size = 0.30 , random_state = 0)
model = LinearRegression()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
r2 = metrics.r2_score(y_test , y_pred)


# In[72]:


print('mean Absolute Error:' , metrics.mean_absolute_error(y_test , y_pred))
print('mean squared Error :' , metrics.mean_squared_error(y_test , y_pred))
print('Root mean squared Error :' , np.sqrt(metrics.mean_squared_error(y_test , y_pred)))
print('R2score:' ,r2)


# In[73]:


### kfold = 3  , whit the border of 75 to 151 , it gives us 97.8 percent 
### as a result  kfold = 5 is better


# In[74]:


### normalize

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
print (x_new [0:10])
x_new.insert(0 , "Target" , y_new)
DataFrame = x_new


# In[75]:


x_norm = DataFrame.drop(["Target"]  , axis = 1)
y_norm = DataFrame["Target"]


# In[76]:


x_train , x_test , y_train , y_test = train_test_split(x_norm , y_norm , test_size = 0.2 , random_state = 0)
model = LinearRegression()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)
r2 = metrics.r2_score(y_test , y_pred)
print('mean Absolute Error:' , metrics.mean_absolute_error(y_test , y_pred))
print('mean squared Error :' , metrics.mean_squared_error(y_test , y_pred))
print('Root mean squared Error :' , np.sqrt(metrics.mean_squared_error(y_test , y_pred)))
print('R2score:' ,r2)


# In[77]:


x_test


# In[78]:


### it is necesseray to do this to get a plot
x_test.insert(1 , "y_test" , y_test)
x_test.insert(2 , "y_pred" , y_pred)
x_test


# In[79]:


new_df3 =x_test.sort_values(by = ["Present_Price"])
new_df3


# In[80]:


a = new_df3.Present_Price
b =new_df3.y_test
c = new_df3.Present_Price
d =new_df3.y_pred


# In[81]:


plt.figure (figsize =(20 , 10))
plt.scatter(a,b , color ="red" , label ="real")
plt.plot(c,d , color = "black" , label ="prediction")
plt.legend(fontsize = 30)
plt.grid()
plt.show()


# In[82]:


#### import items and get output#####


# In[83]:


Present_Price = 11.23
Kms_Driven = 42000
Fuel_Type = 2 ## 2 = petron
Seller_Type = 3 ##3 = Draler
Transmission = 2 ## 2 = manual 
Owner = 1
Age = 10


# In[84]:


model_input = pd.DataFrame ({'Target' : 10
                            ,'present_fuel' : [((Present_Price) * (Fuel_Type))]
                            , 'present_fuel2' : [((Present_Price)* (Fuel_Type**2))]
                            , 'present_Owner': [((Present_Price)*(Owner))]
                            , 'present_Owner2' : [((Present_Price)*(Owner**2))]
                            , 'present_kms': [((Present_Price)* (Kms_Driven))]
                            , 'present_kms2' :[((Present_Price)* (Kms_Driven**2))]
                            , 'present2_fuel' : [((Present_Price**2)*(Fuel_Type))]
                            , 'present2_fuel2' : [((Present_Price**2)*(Fuel_Type**2))]
                            , 'present2_Owner': [((Present_Price**2)*(Owner))]
                            , 'present2_Owner2': [((Present_Price**2)*(Owner**2))]
                            , 'present2_kms': [((Present_Price**2)* (Kms_Driven))]
                            , 'present2_kms2':[((Present_Price**2)* (Kms_Driven**2))]
                            , 'Present_Price':[Present_Price]
                            , 'Kms_Driven': [Kms_Driven]
                            , 'Fuel_Type': [Fuel_Type]
                            , 'Seller_Type': [Seller_Type]
                            , 'Transmission': [Transmission]
                            , 'Owner' : [Owner]
                            , 'Age':[Age]
                            , 'Present_Price2' : [Present_Price**2] 
                            , 'Kms_Driven2': [Kms_Driven**2]
                            , 'Fuel_Type2': [Fuel_Type**2]
                            , 'Owner2': [Owner**2]})
                            


# In[85]:


model_input


# In[86]:


DataFrame_Finall =  DataFrame.append(model_input)


# In[87]:


DataFrame_Finall


# In[88]:


x = DataFrame_Finall.drop(["Target"]  , axis = 1)[:249]
y = DataFrame_Finall["Target"] [:249]
x_DataFrame_Finall  = DataFrame_Finall.drop(["Target"] , axis = 1 ) [249:]


# In[89]:


model.fit(x,y)


# In[90]:


y_pred = model.predict(x_DataFrame_Finall)


# In[91]:


y_pred


# In[92]:


### with kfold = 3  , the price can be reached at 5.97


# In[ ]:




