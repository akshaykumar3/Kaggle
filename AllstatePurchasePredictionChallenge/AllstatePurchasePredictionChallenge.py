import pandas as pd
from matplotlib import pyplot as plt
from pandasql import sqldf

train_file = '/Users/akshaykumar/PycharmProjects/Kaggle/AllstatePurchasePredictionChallenge/data/train.csv'
test_file = '/Users/akshaykumar/PycharmProjects/Kaggle/AllstatePurchasePredictionChallenge/data/test.csv'
solution_file = "/Users/akshaykumar/PycharmProjects/Kaggle/AllstatePurchasePredictionChallenge/data/solution.csv"

train = pd.read_csv(train_file)

# print(train.describe())
# print(train.head())

# print(list(train))
# print(list(train.columns.values))

# first_customer = train.loc[train['customer_ID'] == 10000000]
# print(first_customer)

# last_row = train.loc[train['customer_ID'] in (train['customer_ID'].unique)]

last_row = train.loc[train.groupby('customer_ID').customer_ID.idxmax(), :]
print(last_row)


# figure = plt.figure(figsize=(15,8))
# plt.hist([train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], stacked=True, color = ['g','r'], bins = 30,label = ['Survived','Dead'])
# plt.xlabel('Age')
# plt.ylabel('Number of passengers')
# plt.legend()
# plt.show()
