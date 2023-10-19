#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
import math
import matplotlib.pylab as plt
import seaborn as sns
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='ticks', palette='Set2')
get_ipython().system('pip install graphviz')


# In[3]:


# Load data
path = "./churn2.csv"
df = pd.read_csv(path)[["COLLEGE", "INCOME", "OVERAGE", "LEFTOVER", "HOUSE", "HANDSET_PRICE", "OVER_15MINS_CALLS_PER_MONTH","AVERAGE_CALL_DURATION","REPORTED_SATISFACTION","REPORTED_USAGE_LEVEL","CONSIDERING_CHANGE_OF_PLAN","LEAVE"]].dropna()
#Take a look at the data
df.head(5)


# In[4]:


df.dtypes


# In[5]:


# Transform COLLEGE column to a numeric variable
#df["COLLEGE2"] = (df.COLLEGE =="one").astype(int)
df["COLLEGE2"] = (df.COLLEGE == "one").astype(int)
#df = df.drop("COLLEGE", axis="columns")
#df = df.drop("COLLEGE", axis="columns")
df.head(5)


# In[6]:


df.dtypes


# In[7]:


#df.REPORTED_SATISFACTION = df.REPORTED_SATISFACTION.astype('category')
#df.REPORTED_USAGE_LEVEL = df.REPORTED_USAGE_LEVEL.astype('category')
#df.CONSIDERING_CHANGE_OF_PLAN = df.CONSIDERING_CHANGE_OF_PLAN.astype('category')
#df.COLLEGE2 = df.COLLEGE2.astype('category')
###
df.REPORTED_SATISFACTION = df.REPORTED_SATISFACTION.astype('str')
df.REPORTED_USAGE_LEVEL = df.REPORTED_USAGE_LEVEL.astype('str')
df.CONSIDERING_CHANGE_OF_PLAN = df.CONSIDERING_CHANGE_OF_PLAN.astype('str')


# In[8]:


df["LEAVE2"] = (df.LEAVE=="STAY").astype(int)
#df = df.drop("LEAVE", axis="columns")
df.head(5)


# In[9]:


df.dtypes


# In[10]:


# Names of different columns
predictor_cols = ["INCOME","HOUSE","COLLEGE2"]
target_col ="LEAVE2"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(df[predictor_cols],df[target_col],test_size=0.25,random_state=0)


# In[ ]:





# In[11]:


print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))


# In[34]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
scores=cross_val_score(logreg, X_test ,y_test)
print("Cross validation scores: {}".format(scores))


# In[ ]:





# In[35]:


print("Average cross validation score: {:.2f}".format(scores.mean()))
print("Std. dev of cross validation score: {:.2f}".format(scores.std()))
print("Minimum of cross validation score:{:.2f}".format(scores.min()))


# In[37]:


from sklearn.tree import DecisionTreeClassifier
#Let's define the model (tree)
decision_tree = DecisionTreeClassifier(max_depth=6, criterion="entropy", max_leaf_nodes=12, min_samples_leaf =1)
# let's tell the model what is the data
decision_tree.fit(X_train, y_train)
scores = cross_val_score(decision_tree, X_test, y_test, cv = 10)

print("Cross validation scores: {}".format(scores))
print("Average cross validation score: {:.2f}".format(scores.mean()))
print("Std. dev of cross validation score: {:.2f}".format(scores.std()))
print("Minimum of cross validation score: {:.2f}".format(scores.min()))


# In[38]:


from IPython.display import Image
from sklearn.tree import export_graphviz

def visualize_tree(decision_tree, feature_names, class_names, directory="./images", name="tree", proportion=True):
    #Export our decision tree to graphviz format
    directory1 = directory[2:]
    os.system("mkdir %s" %(directory1))
    dot_name = "%s/%s.dot" % (directory, name)
    dot_file = export_graphviz(decision_tree, out_file=dot_name,feature_names=feature_names,class_names=class_names,
                              proportion=proportion)
    #Call graphviz to make an image file from our decision tree
    image_name = "%s/%s.png" % (directory, name)
    print(dot_name)
    print(image_name)
    #os.system("dot -Tpng %s -o %s" % (dot_name, image_name))
    os.system("dot -Tpng %s -o %s" % (dot_name, image_name))
    # os.system("cd %s" % (directory1))
    #subprocess.run("dot -T png %s -o %s" % (dot_name, image_name))
    #Return the .png image so we can see it
    
    return Image(filename=image_name)

visualize_tree(decision_tree, predictor_cols, ["LEAVE", "STAY"])
    


# In[27]:


y_pred = decision_tree.predict(X_test)
print("Test set score: {: 2f}". format(np.mean(y_pred == y_test)))


# In[28]:


from sklearn import metrics
print ( "Accuracy = %.3f" % (metrics.accuracy_score(decision_tree.predict(X_test), y_test) ))


# In[ ]:





# In[23]:


#predictor_cols = ["INCOME", HOUSE","COLLEGE2"]
X_new = np.array([[700000, 140000,1]])
def Predict_for_New_Value(X_new):
    print("X_new.shape: {}".format(X_new.shape))
    prediction = decision_tree.predict(X_new)
    print("Prediction: {}".format(prediction))
    if(prediction == 0):
        return("LEAVE")
    elif(prediction == 1):
        return("STAY")
    else:
        return("UNKNOWN STATUS..")
    
predicted_status= Predict_for_New_Value(X_new)
print("Predicted value for new record is %s", predicted_status)


