import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# from matplotlib import pyplot as plt


def recover_train_test_target():
    global combined

    train0 = pd.read_csv(train_file)

    targets = train0.SalePrice
    train = combined.head(1460)
    test = combined.iloc[1460:]

    return train, test, targets


def status(feature):
    # print(type(feature))
    print("Processing " + str(feature) + " Ok.")


def get_combined_data():
    # reading train data
    train_data = pd.read_csv(train_file)

    # reading test data
    test_data = pd.read_csv(test_file)

    # extracting and then removing the targets from the training data
    train_data.drop('SalePrice', 1, inplace=True)

    # merging train data and test data for future feature engineering
    combined_data = train_data.append(test_data)
    combined_data.reset_index(inplace=True)
    combined_data.drop('index', inplace=True, axis=1)

    return combined_data


# def map_sale_condition():
#     global combined
#     # # SaleCondition vs Price
#     # c = pd.Series(Counter(data.SaleCondition.str.split(',').sum()))
#     # c.plot(kind='bar', title='Home price')
#     # plt.show()
#
#     # print(list(data['SaleCondition'].unique()))
#     mapping = {'Normal': 1, 'Abnorml': 2, 'Partial': 3, 'AdjLand': 4, 'Alloca': 5, 'Family': 6}
#     # print(combined.SaleCondition.count())
#     combined['SaleCondition'] = combined['SaleCondition'].map(mapping)
#
#     status('SaleCondition')


# def map_sale_type():
#     global combined
#
#     combined.SaleType.fillna('WD', inplace=True)
#     # print(list(combined['SaleType'].unique()))
#     # ['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth']
#     # print(combined.SaleCondition.count())
#     # c = pd.Series(Counter(data.SaleType.str.split(',').sum()))
#     # c.plot(kind='bar', title='Home price')
#     # plt.show()
#     mapping = {'WD': 1, 'New': 2, 'COD': 3, 'ConLD': 4, 'ConLI': 5, 'CWD': 6, 'ConLw': 7, 'Con': 8, 'Oth': 9}
#     combined['SaleType'] = combined['SaleType'].map(mapping)
#
#     status('SaleType')


def map_str_to_numeric(feature):
    global combined

    names = combined[feature].value_counts().index.tolist()
    mapping = {}
    i = 1
    for name in names:
        mapping[name] = i
        i += 1

    # print(combined[feature].value_counts())
    combined[feature] = combined[feature].map(mapping)
    # print(combined[feature].value_counts())
    # status(feature)


def fill_missing_data(feature):
    global combined

    total_count = combined[feature].count()
    names = combined[feature].value_counts().index.tolist()
    series = combined[feature].value_counts()

    prob = []
    size = series.size
    for i in range(size):
        x = series.iloc[i] / total_count
        prob.append(x)

    combined[feature] = combined[feature].fillna(pd.Series(np.random.choice(names, p=prob, size=len(combined))))


train_file = '/Users/akshaykumar/PycharmProjects/Kaggle/HousePricesAdvancedRegressionTechniques/data/train.csv'
test_file = '/Users/akshaykumar/PycharmProjects/Kaggle/HousePricesAdvancedRegressionTechniques/data/test.csv'
solution_file = '/Users/akshaykumar/PycharmProjects/Kaggle/HousePricesAdvancedRegressionTechniques/data/solution.csv'
# data = pd.read_csv(train_file)

combined = get_combined_data()

for y in combined.columns:
    if is_string_dtype(combined[y]):
        fill_missing_data(y)
        map_str_to_numeric(y)

    if combined[y].count() < 2919:
        combined[y].fillna(combined[y].mean(), inplace=True)

train, test, targets = recover_train_test_target()

# print(train['LotFrontage'].count())
# print(train.isnull().values.sum())
# print(train.describe())

# model = linear_model.Lasso(alpha=0.1)
model = linear_model.LassoCV(max_iter=100000000)
# model = linear_model.RidgeCV()
# model = SVR(kernel='rbf', C=1e3, gamma=0.1)
# model = SVR(kernel='linear', C=1e3)
# model = SVR(kernel='poly', C=1e3, degree=2)
# model = DecisionTreeRegressor(max_depth=5)
# model = KernelRidge(alpha=1.0)

model.fit(train, targets)
output = model.predict(test).astype(float)

df_output = pd.DataFrame()

aux = pd.read_csv(test_file)
df_output['Id'] = aux['Id']
df_output['SalePrice'] = output
df_output[['Id', 'SalePrice']].to_csv(solution_file, index=False)
# print(combined.head())

# combined['Fence'] = combined['Fence'].fillna(pd.Series(np.random.choice(['MnPrv', 'GdWo', 'GdPrv', 'MnWw'], p=[0.58, 0.20, 0.20, 0.02], size=len(combined))))
# fill_missing_data('Fence')
# print(list(combined['GarageType'].unique()))
# print(combined['GarageType'].value_counts())

# print(combined.columns.tolist())
# ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
# 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
# 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
# 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
# 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
# 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
# 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

# print(type(combined['Fence'][0]))
# print(combined['Fence'].count())
# print(list(combined['Fence'].unique()))
# print(combined['Fence'].value_counts().index.tolist())
# print(combined['Fence'].value_counts())
