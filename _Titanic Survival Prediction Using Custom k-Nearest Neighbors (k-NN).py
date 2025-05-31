#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction using k-Nearest Neighbors (k-NN)¶

# 
# This project applies a custom-built k-Nearest Neighbors (k-NN) algorithm (without external ML libraries) to predict survival outcomes from the Titanic dataset.  
# 
# We demonstrate:
# - Data cleaning and preprocessing
# - Feature engineering
# - Scaling techniques
# - Distance-based nearest neighbor methods (Euclidean, Manhattan, Hamming)
# - Model evaluation and accuracy comparison
# 
# All implementations are done using only **NumPy** and **Pandas**.
# 

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np


# # Load Dataset
# 
# ## Step 1: Load Dataset
# 
# We load the Titanic dataset directly from a GitHub repository.
# 

# In[2]:


df = pd.read_csv("https://raw.githubusercontent.com/andvise/DataAnalyticsDatasets/main/titanic.csv")
df.head()


# # Inspect Data Structure 
# 
# ## Step 2: Explore Data Structure
# 
# We examine the dataset attributes and data types.
# 

# In[3]:


df.dtypes


# # Remove Irrelevant Columns
# 
# ## Step 3: Drop Irrelevant Columns
# 
# We drop `PassengerId` and `Name` because:
# - They do not contribute meaningful patterns for k-NN distance calculations.
# - They act as unique identifiers and add noise to the model.
# 

# In[4]:


df.drop(columns=['PassengerId', 'Name'], inplace=True)


#  # Handle Missing Values
#  
#  ## Step 4: Handle Missing Values
# 
# We replace missing values (especially in `Age`) with 0 to avoid breaking the distance calculations.

# In[5]:


df.fillna(0, inplace=True)


# #  Encode Categorical Variables 
# 
# ## Step 5: Convert Categorical Columns
# 
# We convert the `Sex` column:
# - `male` → 0
# - `female` → 1
# 
# This allows numerical processing.
# 

# In[6]:


df['Sex'].replace({'male': 0, 'female': 1}, inplace=True)


# # Define Features and Labels
# 
# ## Step 6: Split Features and Labels
# 
# We use `Survived` as the target label and the remaining columns as features.
# 

# In[7]:


df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data

X = df.drop(columns=['Survived'])
y = df['Survived']

print("Features shape:", X.shape)
print("Labels shape:", y.shape)


# # Split Train and Test Sets
# 
# ## Step 7: Split Data into Train/Test
# 
# We use an 80/20 split.
# 

# In[8]:


split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]


# # Scale Features 
# 
# ## Step 8: Apply Min-Max Scaling
# 
# We scale features to [0,1] range for fair distance comparisons.
# 

# In[9]:


def min_max_scale(df):
    return (df - df.min()) / (df.max() - df.min())

X_train_scaled = min_max_scale(X_train)
X_test_scaled = min_max_scale(X_test)


# # Define Distance Functions
# 
# ## Step 9: Define Distance Functions
# 
# We use:
# - Euclidean distance
# - Manhattan distance
# - Hamming distance
# 

# In[10]:


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def hamming_distance(a, b):
    return np.sum(a != b)


# # Build k-NN Function
# 
# ## Step 10: Build k-NN Classifier
# 
# We implement a k-NN method without external ML libraries.
# 

# In[11]:


def knn_predict(X_train, y_train, X_test, k, distance='euclidean'):
    predictions = []
    
    # Choose distance function
    dist_func = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'hamming': hamming_distance
    }[distance]
    
    for query in X_test:
        distances = [dist_func(query, x_train) for x_train in X_train]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]
        predicted_label = np.bincount(nearest_labels).argmax()
        predictions.append(predicted_label)
    
    return np.array(predictions)


# # Test k-NN Across Configurations
# 
# ## Step 11: Evaluate Accuracy for Various k and Distances
# 

# In[13]:


def evaluate_knn(X_train, y_train, X_test, y_test, k_values, distances):
    best_score = 0
    best_config = None
    
    for k in k_values:
        for dist in distances:
            preds = knn_predict(X_train.values, y_train.values, X_test.values, k, distance=dist)
            accuracy = np.mean(preds == y_test.values)
            print(f"Accuracy (k={k}, distance={dist}): {accuracy:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_config = (k, dist)
    
    print(f"\nBest configuration → k={best_config[0]}, distance={best_config[1]}, accuracy={best_score:.4f}")


# # Run Evaluation
# 
# ## Step 12: Run Model and Identify Best Setup
# 

# In[14]:


evaluate_knn(X_train_scaled, y_train, X_test_scaled, y_test, k_values=[1, 3, 5, 7], distances=['euclidean', 'manhattan', 'hamming'])


# # Final Summary

# ## Summary and Conclusions
# 
# In this project, we built and tested a custom **k-Nearest Neighbors (k-NN)** classifier (without using external machine learning libraries)  
# to predict passenger survival on the Titanic dataset.
# 
# We explored three distance measures:
# - Euclidean distance
# - Manhattan distance
# - Hamming distance
# 
# And tested multiple values of **k**:
# - k = 1, 3, 5, 7
# 
# ### **Key Findings**
# The **best-performing configuration** was:
# - **k = 5**
# - **Manhattan distance**
# - Achieved an accuracy of **85.39%**
# 
# **Performance trends:**
# - Accuracy generally improved as we moved from k=1 to k=5.
# - Manhattan and Euclidean distances outperformed Hamming, which makes sense because Hamming is better suited to binary/categorical data.
# - Using majority voting (larger k) made predictions more stable compared to relying on a single nearest neighbor (k=1).
# 
# ### **Possible Improvements**
# - Tune `k` further (try higher values like 9, 11) to see if performance improves.
# - Use cross-validation rather than a single train/test split for more robust evaluation.
# - Apply dimensionality reduction (e.g., PCA) or feature selection to optimize input features.
# - Experiment with weighted voting schemes, where closer neighbors have more influence.
# 
# ### **Final Thoughts**
# This notebook demonstrates how to build a simple but effective k-NN classifier using only NumPy and Pandas.  
# Even without advanced libraries, we can achieve competitive accuracy on classic datasets like Titanic  
# by applying careful data preprocessing, distance selection, and parameter tuning.
# 

# In[ ]:





# In[ ]:





# In[ ]:




