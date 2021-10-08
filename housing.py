import os
import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
'''
housing_path = os.path.join('dataji', 'housing')
os.makedirs(housing_path, exist_ok=True)
print(os.getcwd())

DATA_URL = 'http://www.python.org/ftp/python/2.7.5/Python-2.7.5.tar.bz2'
filename = DATA_URL.split('/')[-1]
print(filename)
'''
HOUSING_PATH = os.path.join('data', 'housing')


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


housing = load_housing_data()
print(housing.info())  # 获取数据集的简单描述，包括总行数、每个属性的类型和非空值的数量
print(housing['ocean_proximity'].value_counts())  # 查看该项数据中有多少种分类存在，每种分类下有多少数量
print(housing.describe())  # 显示数值属性的摘要

#  housing.hist(bins=50, figsize=(10, 15))
#  plt.show()
print(np.random.permutation(9))  # 随机打乱顺序


# 创建测试集
def split_train_test(data, test_ratio):
    '''
    这种情况得到的数据是变化的,两种解决方法，一种是在第一次运行程序后即保存测试集，另一种是在生成随机数之前设置一个随机数生成器
    的种子，从而始终让它生成相同的随机索引
    :param data:
    :param test_ratio:
    :return:

    '''
    np.random.seed(42)  # 设置随机种子为42
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# 创建测试集方法二
housing_with_id1 = housing.reset_index()  # 重置索引，参数drop，为true时，不保留原来的index，否则原来的index作为新的一列
print(housing_with_id1.head())
housing_with_id1['id'] = housing['longitude'] * 1000 + housing['latitude']
# print(crc32(np.int64(123)) & 0xffffffff < 0.2 * 2 ** 32)
# print(crc32(np.int64(0)) & 0xffffffff)
# print(2 ** 32)


# ages = np.array([1, 5, 10, 40, 36, 12, 58, 62, 77, 89, 100, 18, 20, 25, 30, 32])  # 年龄数据
# print(pd.cut(ages, 5, labels=
# 对收入中位数进行类别划分
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# housing['income_cat'].hist()
# plt.show()

# 采用sklearn中的方法实现数据集划分
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# 根据收入类别进行分层抽样(保证比例)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):  # 两个参数分别对应数据和标签
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# 删除多余的属性
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
# 进一步观察数据,首先创建一个训练数据的副本，以确保不破坏原始数据
housing_backup = strat_train_set.copy()
housing_backup.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing_backup['population']/100,
                    label='population', figsize=(10, 7), c='median_house_value', cmap=plt.get_cmap('hot'),
                    colorbar=True)
# plt.show()
# 根据皮尔逊系数寻找数据每对属性之间的相关性
corr_matrix = housing_backup.corr()
print(corr_matrix)
print(corr_matrix['longitude'][:].shape)

data = [['Google', 10], ['Runoob', 2], ['Wiki', 13]]
df = pd.DataFrame(data, index=['aca', 'basd', 'casd'], columns=['Site', 'Age'], dtype=float)
print(df.loc['aca'])