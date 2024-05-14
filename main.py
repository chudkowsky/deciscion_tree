from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

from decision_tree import DecisionTree

# Load iris dataset
data = datasets.load_iris()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Create a Decision Tree classifier with a maximum depth of 10
clf = DecisionTree(max_depth=10)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test)

# Define a function to calculate accuracy
def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

# Calculate the accuracy of the model
acc = accuracy(y_test, predictions)
# Print the accuracy
print(acc)
