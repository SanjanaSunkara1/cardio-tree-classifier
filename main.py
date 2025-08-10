from os import X_OK
import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows
print(df.head())

input("\n Press Enter to continue.\n")

#Data Cleaning
#Label encode the dataset
df = util.labelEncoder(df, [
    "HeartDisease", "GenHealth", "Smoking", "AlcoholDrinking",
    "Sex", "PhysicalActivity"
])

print("\nHere is a preview of the dataset after label encoding. \n")
print(df.head())

input("\nPress Enter to continue.\n")

#One hot encode the dataset
df = util.oneHotEncoder(df, ["Race", "AgeCategory"])
print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(df.head())
input("\nPress Enter to continue.\n")

#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 9, class_weight = "balanced")
clf = clf.fit(X_train, y_train)

#Test the model with the testing data set and prints accuracy score
test_predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test,test_predictions)
print("The accuracy with the testing data set of the Decision Tree is: " + str(test_acc))

#Prints the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, test_predictions, labels = [1,0])
print("The confusion matrix of the tree is: ")
print(cm)

#Test the model with the training data set and prints accuracy score
train_predictions = clf.predict(X_train)

from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train,train_predictions)
print("The accuracy with the training data set of the Decision Tree is: " + str(train_acc))
input("\nPress Enter to continue.\n")

#Prints another application of Decision Trees and considerations
print(
    "\nBelow is another application of decision trees and considerations for using them:\n"
)

print(
    "\nDecision Trees can be used in many fields beyond healthcare. For example, in sports, they can help predict player performance or the outcome of games based on player stats and past results. "
    "When building these models, it is important to consider factors like data quality, avoiding bias in the dataset, and making sure the model generalizes well to new data (not just memorizing training data).\n"
)

#Prints a text representation of the Decision Tree
print(
"\nBelow is a text representation of how the Decision Tree makes choices:\n"
)

print("Decision trees make choices by asking a series of yes or no questions about the data. At each step, the tree looks at one feature and decides the best way to split the data into groups. It keeps splitting until it can separate the data into groups that are mostly one outcome. This helps the tree decide the final answer by following the path of questions down to a decision.")

input("\nPress Enter to continue.\n")
util.printTree(clf, X.columns)

#Prints how a Decision Tree can be used in another field
print("\n---\n")
print("How can a Decision Tree be used in a field I’m passionate about?\n")

print("I'm passionate about movies. A Decision Tree could help build a movie recommendation system. It could classify what kind of movie a user would like based on their past preferences, age, genre interests, and ratings given to previous films.")

print("\nWhat factors should I be careful about when creating the Decision Tree?\n")
print("Representation: Make sure the training data includes people of different ages, backgrounds, and preferences.")
print("Label Encoding and One Hot Encoding: Ensure categorical features like 'genre' or 'rating' are properly encoded.")
print("Randomness and Bias: Be cautious of overfitting. If the model memorizes the training set, it won’t work well for new users.")
print("Weight: If some movie genres are more popular in the dataset, balance the model so it doesn't unfairly favor those genres.")
