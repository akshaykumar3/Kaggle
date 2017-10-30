import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


def sanitize_data():
    global combined

    size = len(combined.index)

    for y in combined.columns:
        if is_string_dtype(combined[y]):
            fill_missing_data(y)
            map_str_to_numeric(y)

        if combined[y].count() < size:
            combined[y].fillna(combined[y].mean(), inplace=True)


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
solution_file = '/Users/akshaykumar/PycharmProjects/Kaggle/HousePricesAdvancedRegressionTechniques/data/solution2.csv'

train = pd.read_csv(train_file)
targets = train.SalePrice

# extracting and then removing the targets from the training data
train.drop('SalePrice', 1, inplace=True)

combined = train
sanitize_data()
train = combined

test = pd.read_csv(test_file)

pd.read_excel()

combined = test
sanitize_data()
test = combined



pd.melt()


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