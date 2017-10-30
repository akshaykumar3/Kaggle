import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
import numpy as np

# remove warnings
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = 100


def status(feature):
    print("Processing " + str(feature) + " ok.")


def get_combined_data():
    # reading train data
    train_data = pd.read_csv(train_file)

    # reading test data
    test_data = pd.read_csv(test_file)

    # extracting and then removing the targets from the training data
    targets = train_data.Survived
    train_data.drop('Survived', 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined_data = train_data.append(test_data)
    combined_data.reset_index(inplace=True)
    combined_data.drop('index', inplace=True, axis=1)

    return combined_data


# Fill missing Embark value with most frequent Embark value.
def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.head(891).Embarked.fillna('S', inplace=True)
    combined.iloc[891:].Embarked.fillna('S', inplace=True)

    # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)

    status('embarked')


# To fill the missing fare value with mean
def process_fares():
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)
    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace=True)

    status('fare')


def get_titles():
    global combined

    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated titles
    title_dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }

    # we map each title
    combined['Title'] = combined.Title.map(title_dictionary)
    # print(combined.Title)


# Drop the feature Name and keep only Titles.
def process_names():
    global combined
    # we clean the Name variable
    combined.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)

    # removing the title variable
    combined.drop('Title', axis=1, inplace=True)

    status('names')


# Gets the median age based on other attributes.
def process_age():
    global combined

    # a function that fills the missing values of the Age variable

    def fill_ages(row, grouped_median):
        if row['Sex'] == 'female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female', 1, 'Royalty']['Age']

        elif row['Sex'] == 'female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']

        elif row['Sex'] == 'female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Officer']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']

    combined.head(891).Age = combined.head(891).apply(lambda r: fill_ages(r, grouped_median_train) if np.isnan(r['Age'])
    else r['Age'], axis=1)

    combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r: fill_ages(r, grouped_median_test) if np.isnan(r['Age'])
    else r['Age'], axis=1)

    status('age')


# This function replaces NaN values with U (for Unknown).
# It then maps each Cabin value to the first letter. Then it encodes the cabin values using dummy encoding again.
def process_cabin():
    global combined

    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U', inplace=True)

    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')

    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)

    status('cabin')


# Assign 1 to male and 0 to female
def process_sex():
    global combined
    # mapping string values to numerical one
    combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})

    status('sex')


# Encoding into 3 categories
def process_pclass():
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    # adding dummy variables
    combined = pd.concat([combined, pclass_dummies], axis=1)

    # removing "Pclass"

    combined.drop('Pclass', axis=1, inplace=True)

    status('pclass')


# This functions pre process the tickets first by extracting the ticket prefix.
# When it fails in extracting a prefix it returns XXX.
# Then it encodes prefixes using dummy encoding.
def process_ticket():
    global combined

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def clean_ticket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = list(filter(lambda t: not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(clean_ticket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)

    status('ticket')


# This function introduces 4 new features:
# FamilySize : the total number of relatives including the passenger (him/her)self.
# Singleton : a boolean variable that describes families of size = 1
# SmallFamily : a boolean variable that describes families of 2 <= size <= 4
# LargeFamily : a boolean variable that describes families of 5 < size
def process_family():
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    status('family')


def recover_train_test_target():
    global combined

    train0 = pd.read_csv(train_file)

    targets = train0.Survived
    train = combined.head(891)
    test = combined.iloc[891:]

    return train, test, targets


def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv=5, scoring=scoring)
    return np.mean(xval)


train_file = '/Users/akshaykumar/PycharmProjects/Kaggle/Titanic/data/train.csv'
test_file = '/Users/akshaykumar/PycharmProjects/Kaggle/Titanic/data/test.csv'
solution_file = "/Users/akshaykumar/PycharmProjects/Kaggle/Titanic/data/solution1.csv"

data = pd.read_csv(train_file)
# test = pd.read_csv(test_file)

# print(train.describe())
# print(train.head())

# Fill the missing ages
data['Age'].fillna(data['Age'].median(), inplace=True)
# print(train.describe())

# # Survival based on Sex
survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))
plt.show()
#
# # Survival based on Age
figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'], bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()
#
# # Survival based on Fare
# figure = plt.figure(figsize=(15,8))
# plt.hist([train[train['Survived']==1]['Fare'],train[train['Survived']==0]['Fare']],
# stacked=True, color = ['g','r'], bins = 30,label = ['Survived','Dead'])
# plt.xlabel('Fare')
# plt.ylabel('Number of passengers')
# plt.legend()
#
#
# # Survival based on Age and Fare combined
# plt.figure(figsize=(15,8))
# ax = plt.subplot()
# ax.scatter(train[train['Survived']==1]['Age'],train[train['Survived']==1]['Fare'],c='green',s=40)
# ax.scatter(train[train['Survived']==0]['Age'],train[train['Survived']==0]['Fare'],c='red',s=40)
# ax.set_xlabel('Age')
# ax.set_ylabel('Fare')
# ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)

# # Fare Vs Class
# ax = plt.subplot()
# ax.set_ylabel('Average fare')
# train.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)


# # Survival based on Embarkation
# survived_embark = train[train['Survived'] == 1]['Embarked'].value_counts()
# dead_embark = train[train['Survived'] == 0]['Embarked'].value_counts()
# df = pd.DataFrame([survived_embark, dead_embark])
# df.index = ['Survived', 'Dead']
# df.plot(kind='bar', stacked=True, figsize=(15, 8))
# plt.show()










# Getting combined train and test data.
combined = get_combined_data()

# Get the Titles
get_titles()
# print(combined.head())

# Get Median age of grouped data for both training and test set.
grouped_train = combined.head(891).groupby(['Sex', 'Pclass', 'Title'])
grouped_median_train = grouped_train.median()

grouped_test = combined.iloc[891:].groupby(['Sex', 'Pclass', 'Title'])
grouped_median_test = grouped_test.median()
# print(grouped_median_train)

# process_age()
combined["Age"] = combined.groupby(['Sex', 'Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))
process_names()
process_fares()
process_embarked()
process_cabin()
# No more missing values.


process_sex()
process_pclass()
process_ticket()
process_family()

# Let's remove passengerId as it conveys no information.
combined.drop('PassengerId', inplace=True, axis=1)
# print(combined.info())

train, test, targets = recover_train_test_target()

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(20, 20))
# plt.show()

model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
print(train_reduced.shape)

test_reduced = model.transform(test)
print(test_reduced.shape)

# turn run_gs to True if you want to run the Grid Search again.
run_gs = False

if run_gs:
    parameter_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [1, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(targets, n_splits=5)

    grid_search = GridSearchCV(forest, scoring='accuracy', param_grid=parameter_grid, cv=cross_validation)

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else:
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50,
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)

# print(compute_score(model, train, targets, scoring='accuracy'))

output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv(test_file)
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId', 'Survived']].to_csv(solution_file, index=False)
