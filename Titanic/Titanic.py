# https://www.kaggle.com/c/titanic
# import csv
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np

train_file = '/Users/akshaykumar/PycharmProjects/Kaggle/Titanic/data/train.csv'
test_file = '/Users/akshaykumar/PycharmProjects/Kaggle/Titanic/data/test.csv'
solution_file = "/Users/akshaykumar/PycharmProjects/Kaggle/Titanic/data/solution.csv"

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# Clean Training data
# train["Child"] = float('NaN')
# train["Child"][train["Age"] < 18] = 1
# train["Child"][train["Age"] >= 18] = 0

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# train["Age"][train["Age"] == float('NaN')] = train.Age.median()
train["Age"].fillna(train.Age.median(), inplace=True)
# print(train)

target = train["Survived"].values

# Build the tree model
# features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
#
# my_tree_one = tree.DecisionTreeClassifier()
# my_tree_one = my_tree_one.fit(features_one, target)


# Build the random forest model
# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
train["SibSp"].fillna(train.SibSp.median(), inplace=True)
train["Parch"].fillna(train.Parch.median(), inplace=True)
train["Embarked"].fillna(train.Embarked.median(), inplace=True)
print(train)
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
my_forest = forest.fit(features_forest, target)


# Clean test data
test.Fare[152] = test.Fare.median()

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

test["Age"].fillna(test.Age.median(), inplace=True)
test["SibSp"].fillna(test.SibSp.median(), inplace=True)
test["Parch"].fillna(test.Parch.median(), inplace=True)
test["Embarked"].fillna(test.Embarked.median(), inplace=True)


# Extract the features from the test set: Pclass, Sex, Age, and Fare.
# test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features)

# Make your prediction using the test set and print them.
# my_prediction = my_tree_one.predict(test_features)
# print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns=["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv(solution_file, index_label=["PassengerId"])
