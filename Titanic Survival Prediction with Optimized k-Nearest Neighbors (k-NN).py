#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction with Custom k-Nearest Neighbors (k-NN)
# 
# This project builds a custom k-Nearest Neighbors (k-NN) classifier  
# to predict survival outcomes on the Titanic dataset.
# 
# We walk through:
# - Data loading and preparation
# - Feature transformation
# - Data scaling
# - Train-test splitting
# - Preparing the dataset for distance-based classification models
# 
# All steps are fully documented, with no external machine learning libraries used  
# (only NumPy and Pandas).
# 

# ## Step 1: Load the Titanic Dataset
# 
# We load the Titanic dataset directly from a GitHub-hosted CSV file  
# using Pandas, which gives us a structured DataFrame to work with.
# 

# In[1]:


# Importing necessary libraries
import pandas as pd       # Pandas for data manipulation and table structures
import numpy as np        # NumPy for numerical operations and array handling

# Load the Titanic dataset from an online GitHub link into a pandas DataFrame
df = pd.read_csv("https://raw.githubusercontent.com/andvise/DataAnalyticsDatasets/main/titanic.csv")

# Display the first few rows to get an initial look at the dataset
df.head()


# # Check Attribute Types
# 
# ## Step 2: Explore Dataset Attributes
# 
# We check the names and data types of all columns  
# to understand which variables are numeric, categorical, or text-based.
# 

# In[2]:


# Display the names and data types of each column in the DataFrame
df.dtypes

# This helps identify:
# - Numerical vs. categorical variables
# - Which columns might need transformation or removal


# # Remove Irrelevant Columns
# 
# ## Step 3: Remove Irrelevant Columns
# 
# We remove columns like `PassengerId` and `Name` because:
# - They are identifiers, not predictive features.
# - They don’t provide meaningful patterns for k-NN distance calculations.
# 

# In[3]:


# Drop 'PassengerId' and 'Name' as they are unique identifiers and not useful for modeling
df.drop(columns=['PassengerId', 'Name'], inplace=True)

# Confirm the remaining columns
df.head()


# # Handle Missing Values 
# 
# ## Step 4: Replace Missing Values
# 
# To ensure consistent numerical computations,  
# we replace any missing values (NaNs) with `0`.
# 

# In[4]:


# Replace missing values (NaN) in the DataFrame with 0
df.fillna(0, inplace=True)

# Check if any missing values remain (should return all False)
df.isnull().any()


# # Convert Categorical Columns 
# 
# ## Step 5: Transform Categorical Columns
# 
# We convert the `Sex` column:
# - `male` → 0
# - `female` → 1
# 
# This allows the algorithm to process it as a numeric feature.
# 

# In[5]:


# Replace 'male' with 0 and 'female' with 1 in the 'Sex' column
df['Sex'].replace({'male': 0, 'female': 1}, inplace=True)

# Confirm the transformation
df.head()


# # Define Features and Labels 
# 
# ## Step 6: Split Features and Target
# 
# We separate:
# - **Features (`X`)** → all columns except `Survived`
# - **Target (`y`)** → the `Survived` column

# In[6]:


# Shuffle the DataFrame rows randomly to prevent order bias
df = df.sample(frac=1).reset_index(drop=True)

# Split into features (X) and target labels (y)
x = df.drop(columns=['Survived'])    # Features
y = df['Survived']                   # Target label

# Display the shapes
print("Shape of features matrix (x):", x.shape)
print("Shape of target labels (y):", y.shape)


# # Train-Test Split
# 
# ## Step 7: Split into Training and Test Sets
# 
# We divide the data:
# - 80% for training
# - 20% for testing
# 

# In[7]:


# Calculate the split index for 80% training data
samples = int(len(df) * 0.8)

# Create training and test sets
train_set = x.iloc[:samples]     # First 80% for training
test_set = x.iloc[samples:]      # Remaining 20% for testing

# Separate corresponding target labels
train_label = y.iloc[:samples]
test_label = y.iloc[samples:]

# Confirm the splits
print("Training set shape:", train_set.shape)
print("Test set shape:", test_set.shape)
print("Training labels shape:", train_label.shape)
print("Test labels shape:", test_label.shape)


# # Apply Min-Max Scaling
# 
# ## Step 8: Scale Features with Min-Max Normalization
# 
# We normalize all features to a [0, 1] range  
# so that no single feature dominates the distance calculations.
# 

# In[8]:


# Define a function for min-max scaling
def min_max_scaling(scaled):
    # Scale each feature column to range [0,1]
    return (scaled - scaled.min()) / (scaled.max() - scaled.min())

# Apply scaling to both train and test sets
train_set_scaled = min_max_scaling(train_set)
test_set_scaled = min_max_scaling(test_set)

# Display the first few scaled rows
print("Scaled Training Dataset:\n", train_set_scaled.head())
print("\nScaled Testing Dataset:\n", test_set_scaled.head())


# # Check Final Shapes
# 
# ## Step 9: Confirm Final Dataset Shapes
# 
# We verify that our scaled datasets are ready  
# with matching shapes between features and labels.
# 

# In[9]:


# Print the shapes of the scaled training and testing sets
print("Shape of Scaled Training Set:", train_set_scaled.shape)
print("Shape of Scaled Testing Set:", test_set_scaled.shape)

# Print the shapes of the corresponding label arrays
print("Shape of training labels:", train_label.shape)
print("Shape of testing labels:", test_label.shape)


# # Nearest Neighbor (NN) Model
# 
# ## Step 10: Implement Nearest Neighbor (NN) Model
# 
# We implement a simple nearest neighbor (NN) classifier using Euclidean distance.
# 
# For each test point:
# - We find the training point with the **smallest Euclidean distance**.
# - We assign the label of this nearest training point as the prediction.
# 
# Finally, we compute and report the accuracy of this model.
# 

# # Define Euclidean Distance

# In[10]:


# Define a function to calculate Euclidean distance between two points (arrays)
def euclidean_distance(index1, index2):
    # Square the differences in each dimension, sum them, then take the square root
    return np.sqrt(np.sum((index1 - index2) ** 2))


# # Make Predictions Using NN

# In[12]:


# Define a function to predict labels for the test set using NN

def find_nn_predictions(train_set_scaled, train_label, test_set_scaled):
    predicted = []  # Initialize an empty list to store predictions

    # Iterate over each sample in the test set
    for query in test_set_scaled:
        # Calculate Euclidean distance between the query and all training samples
        distances = [euclidean_distance(query, train_sample) for train_sample in train_set_scaled]
        
        # Find the index of the nearest neighbor (minimum distance)
        nearest_neighbor_index = np.argmin(distances)
        
        # Assign the label of the nearest neighbor as the prediction
        predicted_label = train_label[nearest_neighbor_index]
        predicted.append(predicted_label)
    
    # Return the list of predicted labels as a NumPy array
    return np.array(predicted)

# Run the NN predictions on the test set
predictions_nn = find_nn_predictions(train_set_scaled.values, train_label.values, test_set_scaled.values)


# # Compare Predictions and Actual Labels

# In[13]:


# Create a DataFrame to display both predicted and actual labels for easy comparison
prediction_df = pd.DataFrame({
    'Predicted': predictions_nn,
    'Actual': test_label.values
})

# Display the first few rows to check
print("Prediction DataFrame:\n", prediction_df.head())


# #  Compute Accuracy

# In[14]:


# Calculate the number of correct predictions
accurate_predictions = (prediction_df['Predicted'] == prediction_df['Actual']).sum()

# Calculate the accuracy as the proportion of correct predictions
accuracy = accurate_predictions / len(prediction_df)

# Print the final accuracy result
print(f"Accuracy of Nearest Neighbor (NN) model: {accuracy:.4f}")


# # Summary of NN Model
# 
# The Nearest Neighbor (NN) model achieved:
# - **Accuracy ≈ 72.47%**
# 
# This means the model correctly classified about 72 out of every 100 passengers.
# 
# Key observations:
#  The model is simple and relies only on the **single closest neighbor**.  
#  While it provides a reasonable baseline, its predictions can be sensitive to noise or outliers.
# 
# This sets the stage for improving performance using:
# - **k-Nearest Neighbors (k-NN)**, which averages over multiple neighbors.
# - **Different distance metrics** (e.g., Manhattan, Hamming) to better capture data relationships.

# # k-Nearest Neighbors (k-NN) Model 
# 
# ## Step 11: Implement k-Nearest Neighbors (k-NN) Model
# 
# We extend the previous nearest neighbor approach by using **k neighbors**  
# to make predictions based on the majority vote among the k closest points.
# 
# Additionally, we implement:
# - Euclidean distance
# - Manhattan distance
# - Hamming distance
# 
# We will compute and compare accuracies across these distance measures.
# 

# # Define Distance Functions

# In[15]:


# Define Euclidean distance between two points (arrays)
def euclidean_distance(index1, index2):
    return np.sqrt(np.sum((index1 - index2) ** 2))

# Define Manhattan distance (sum of absolute differences)
def manhattan_distance(index1, index2):
    return np.sum(np.abs(index1 - index2))

# Define Hamming distance (number of differing elements)
def hamming_distance(index1, index2):
    return np.sum(index1 != index2)


# # Build k-NN Prediction Function

# In[16]:


# Define k-NN function: predicts labels based on majority vote among k neighbors

def find_knn_predictions(train_set, train_label, test_set, k, distance='euclidean'):
    predictions = []  # Store predictions
    
    # Select appropriate distance function
    if distance == 'euclidean':
        distance_method = euclidean_distance
    elif distance == 'manhattan':
        distance_method = manhattan_distance
    elif distance == 'hamming':
        distance_method = hamming_distance
    
    # Iterate over each test point
    for query in test_set:
        # Calculate distances to all training points
        distances = [distance_method(query, train_point) for train_point in train_set]
        
        # Get indices of k closest neighbors
        nearest_neighbors = np.argsort(distances)[:k]
        
        # Retrieve their labels
        neighbor_labels = train_label[nearest_neighbors]
        
        # Determine the most frequent label (majority vote)
        predicted = np.bincount(neighbor_labels).argmax()
        
        predictions.append(predicted)
    
    return np.array(predictions)


# # Evaluate k-NN Accuracy (k=4)

# In[17]:


# Run k-NN with k=4 using each distance metric

knn_predictions_euclidean = find_knn_predictions(
    train_set_scaled.values, train_label.values, test_set_scaled.values, 4, distance='euclidean')

knn_predictions_manhattan = find_knn_predictions(
    train_set_scaled.values, train_label.values, test_set_scaled.values, 4, distance='manhattan')

knn_predictions_hamming = find_knn_predictions(
    train_set_scaled.values, train_label.values, test_set_scaled.values, 4, distance='hamming')

# Compute accuracy for each metric
euclidean_accuracy = (knn_predictions_euclidean == test_label.values).mean()
manhattan_accuracy = (knn_predictions_manhattan == test_label.values).mean()
hamming_accuracy = (knn_predictions_hamming == test_label.values).mean()

# Print results
print(f"Accuracy (Euclidean Distance): {euclidean_accuracy:.4f}")
print(f"Accuracy (Manhattan Distance): {manhattan_accuracy:.4f}")
print(f"Accuracy (Hamming Distance): {hamming_accuracy:.4f}")


# # Accuracy Comparison
# 
# We compare the performance of k-NN (k=4) across three distance measures:
# - **Euclidean**
# - **Manhattan**
# - **Hamming**
# 
# We will check which distance metric offers better predictive power  
# compared to the simple NN model (single neighbor).
# 

# ### Is k-NN More Accurate Compared to NN?
# 
# From the comparison, we observe that the k-NN implementation indeed improves accuracy compared to the simpler NN (single nearest neighbor) approach.
# 
# Specifically:
# - The NN model achieved an accuracy of **72.47%**.
# - The k-NN model (k=4) achieved:
#     - **Euclidean distance:** 82.58%
#     - **Manhattan distance:** 83.15% (highest)
#     - **Hamming distance:** 78.65%
# 
#  **Key takeaways:**
# - Both Euclidean and Manhattan k-NN outperform the simple NN approach by ~10 percentage points.
# - Manhattan distance slightly outperforms Euclidean, suggesting that absolute difference-based comparisons work better for this dataset than squared-difference comparisons.
# - Hamming distance lags behind, which is expected, as it is more suitable for binary/categorical data and less sensitive to continuous numerical differences.
# 
# Thus, the k-NN approach offers better generalization, reduces sensitivity to outliers, and strengthens predictive accuracy  
# by leveraging multiple neighbors rather than relying on a single closest point.
# 

# # Test Multiple k and Distance Combinations

# In[18]:


# Test multiple combinations of k and distance metrics

def test_knn(train_set, train_label, test_set, test_label):
    k_values = [1, 3, 5, 7]
    distance_methods = ['euclidean', 'manhattan', 'hamming']
    
    top_accuracy = 0
    best_combination = None
    
    # Iterate over each k and distance combination
    for k in k_values:
        for distance in distance_methods:
            predictions = find_knn_predictions(train_set, train_label, test_set, k, distance)
            accuracy = np.mean(predictions == test_label)
            print(f"Accuracy (k={k}, distance={distance}): {accuracy:.4f}")
            
            # Track best combination
            if accuracy > top_accuracy:
                top_accuracy = accuracy
                best_combination = (k, distance)
    
    print(f"\nBest combination → k={best_combination[0]}, distance={best_combination[1]}, accuracy={top_accuracy:.4f}")

# Run the full test
test_knn(train_set_scaled.values, train_label.values, test_set_scaled.values, test_label.values)


# # Summary of Results

# ## Final Summary: Titanic Survival Prediction with k-NN
# 
# In this project, we implemented and evaluated a custom **k-Nearest Neighbors (k-NN)** classifier  
# (without external ML libraries) on the Titanic dataset to predict passenger survival.
# 
# ### Key Findings:
# - **Best configuration:**  
#     - k = 5  
#     - Distance metric = Euclidean  
#     - Achieved accuracy ≈ **84.83%**
#   
# - **Performance trends:**
#     - Increasing k from 1 to 5 improved accuracy, as majority voting smoothed out noise.
#     - Euclidean distance slightly outperformed Manhattan distance.
#     - Hamming distance, while acceptable, generally underperformed compared to continuous distance measures due to the mixed nature of the dataset (both continuous and categorical features).
# 
# ### Best Result:
# | Configuration        | Accuracy  |
# |----------------------|-----------|
# | k = 5, Euclidean     | 84.83%   |
# 
# ### Next Steps:
# To further enhance performance, we recommend:
# - Exploring advanced hyperparameter tuning.
# - Testing weighted distance approaches.
# - Applying dimensionality reduction.
# - Trying alternative machine learning models.
# 
# This project demonstrates that even a simple, handcrafted k-NN approach can deliver competitive performance on classic datasets  
# when supported by careful preprocessing, parameter selection, and distance metric design.
# 

# ### How Can We Improve the Results?
# 
# Based on the current best performance (k=5, Euclidean distance, accuracy ≈ 84.83%),  
# we can explore the following strategies to push the results even further:
# 
# **Hyperparameter Tuning:**  
# Experiment with additional k values (e.g., 9, 11, 13) and alternative distance metrics or weightings (such as weighted k-NN, where closer neighbors have more influence).
# 
# **Advanced Preprocessing:**  
# Apply feature transformations, standardization (z-score scaling), or outlier removal to improve the data quality and make distances more meaningful.
# 
# **Dimensionality Reduction:**  
# Use techniques like Principal Component Analysis (PCA) or feature selection to reduce the feature space, remove redundant variables, and improve computational efficiency.
# 
# **Algorithm Exploration:**  
# Compare k-NN with other classification models such as logistic regression, decision trees, or random forests to assess whether another algorithm offers better predictive performance.
# 
# **Cross-Validation:**  
# Instead of relying on a single train/test split, perform k-fold cross-validation to get more robust estimates of model performance and reduce variability.
# 

# In[ ]:
