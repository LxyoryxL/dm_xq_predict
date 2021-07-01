from operator import pos
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import imblearn
from palettable.colorbrewer.qualitative import Pastel1_3

df = pd.read_csv('./dm.csv', encoding='gbk')
df_test = pd.read_csv('./dm_test.csv', encoding='gbk')

# 数据整合
x = df[['x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']]
y = df['y']
print(x.shape, y.shape)

X_test = df_test[[
    'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'
]]
y_test = df_test['y']
print(X_test.shape, y_test.shape)

# 做训练集和测试集分割
X_train, X_valid, y_train, y_valid = train_test_split(x,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=0,
                                                      stratify=y)
print(X_train.shape, X_valid.shape)

# 针对训练集，过采样
# oversample = imblearn.over_sampling.SVMSMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train)

# 决策树
dtc = DecisionTreeClassifier()

# 交叉验证
print("决策树预测出的验证准确率为：{}".format(cross_val_score(dtc, X_train, y_train, cv=5)))
print(
    '------------------------------------------------------------------------------------'
)

dtc.fit(X_train, y_train)

# 验证
# pre_valid_dtc = dtc.predict(X_valid)

# print("决策树预测出的验证准确率为：{}".format(
#     metrics.accuracy_score(y_true=y_valid, y_pred=pre_valid_dtc)))
# print("决策树预测出的验证recall为：{}".format(
#     metrics.recall_score(y_true=y_valid, y_pred=pre_valid_dtc)))
# print("决策树预测出的验证precision为：{}".format(
#     metrics.precision_score(y_true=y_valid, y_pred=pre_valid_dtc)))
# print("决策树预测出的验证f1为：{}".format(
#     metrics.f1_score(y_true=y_valid, y_pred=pre_valid_dtc)))
# print(
#     '------------------------------------------------------------------------------------'
# )

# 测试
pre_test = (dtc.predict_proba(X_test)[:, 1] >= 0.2).astype(int)

print(dtc.predict_proba(X_test))

# pre_test = dtc.predict(X_test)
print(pre_test)
print(np.where(pre_test == 1))
print(np.where(y_test == 1))
print("决策树预测出的测试准确率为：{}".format(
    metrics.accuracy_score(y_true=y_test, y_pred=pre_test)))
print("决策树预测出的测试recall为：{}".format(
    metrics.recall_score(y_true=y_test, y_pred=pre_test)))
print("决策树预测出的测试precision为：{}".format(
    metrics.precision_score(y_true=y_test, y_pred=pre_test)))
print("决策树预测出的测试f1为：{}".format(metrics.f1_score(y_true=y_test,
                                                y_pred=pre_test)))
print("混淆矩阵：{}".format(confusion_matrix(y_test, pre_test)))
print(
    '------------------------------------------------------------------------------------'
)

# 神经网络
mlp = MLPClassifier(random_state=2, max_iter=3000).fit(X_train, y_train)

# 交叉验证
print("神经网络预测出的验证准确率为：{}".format(cross_val_score(mlp, X_train, y_train, cv=5)))
print(
    '------------------------------------------------------------------------------------'
)

mlp.fit(X_train, y_train)

# 验证
# pre_valid_mlp = mlp.predict(X_valid)
# print("神经网络预测出的验证acc为：{}".format(metrics.accuracy_score(
#     y_valid, pre_valid_mlp)))
# print("神经网络预测出的验证recall为：{}".format(
#     metrics.recall_score(y_valid, pre_valid_mlp)))
# print("神经网络预测出的验证precision为：{}".format(
#     metrics.precision_score(y_valid, pre_valid_mlp)))
# print("神经网络预测出的验证f1为：{}".format(metrics.f1_score(y_valid, pre_valid_mlp)))
# print(
#     '------------------------------------------------------------------------------------'
# )

# 测试
pre_test_mlp = mlp.predict(X_test)
print(pre_test_mlp)
print(np.where(pre_test_mlp == 1))
print(np.where(y_test == 1))
print("神经网络预测出的测试acc为：{}".format(metrics.accuracy_score(y_test, pre_test_mlp)))
print("神经网络预测出的测试recall为：{}".format(metrics.recall_score(y_test,
                                                         pre_test_mlp)))
print("神经网络预测出的测试precision为：{}".format(
    metrics.precision_score(y_test, pre_test_mlp)))
print("神经网络预测出的测试f1为：{}".format(metrics.f1_score(y_test, pre_test_mlp)))
print("混淆矩阵：{}".format(confusion_matrix(y_test, pre_test_mlp)))
