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

df_train = pd.read_csv('./speed_dating_train.csv', encoding='gbk')
df_test = pd.read_csv('./speed_dating_test.csv', encoding='gbk')

# 根据热力图、缺失率选择一些特征
# date_df = df_train[[
#     'like', 'like_o', 'fun', 'fun_o', 'attr', 'prob', 'attr_o', 'prob_o',
#     'intel', 'sinc', 'intel_o', 'sinc_o', 'clubbing', 'fun1_1', 'fun3_1',
#     'pf_o_fun', 'attr3_1', 'art', 'met', 'yoga', 'concerts', 'dining', 'music',
#     'int_corr', 'pf_o_att', 'gaming', 'intel1_1', 'hiking', 'sports',
#     'reading', 'museums', 'samerace', 'pf_o_int', 'attr1_1', 'gender',
#     'sinc3_1', 'tvsports', 'match', 'field_cd', 'career_c'
# ]]

# dating_test = df_test[[
#     'like', 'like_o', 'fun', 'fun_o', 'attr', 'prob', 'attr_o', 'prob_o',
#     'intel', 'sinc', 'intel_o', 'sinc_o', 'clubbing', 'fun1_1', 'fun3_1',
#     'pf_o_fun', 'attr3_1', 'art', 'met', 'yoga', 'concerts', 'dining', 'music',
#     'int_corr', 'pf_o_att', 'gaming', 'intel1_1', 'hiking', 'sports',
#     'reading', 'museums', 'samerace', 'pf_o_int', 'attr1_1', 'gender',
#     'sinc3_1', 'tvsports', 'match','field_cd', 'career_c'
# ]]

date_df = df_train[[
    'match', 'int_corr', 'samerace', 'age_o', 'race_o', 'pf_o_att', 'pf_o_sin',
    'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha', 'attr_o', 'sinc_o',
    'intel_o', 'fun_o', 'like_o', 'prob_o', 'met_o', 'age', 'race', 'imprace',
    'imprelig', 'goal', 'date', 'go_out', 'career_c', 'sports', 'tvsports',
    'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing',
    'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping',
    'yoga', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'attr3_1',
    'sinc3_1', 'fun3_1', 'intel3_1', 'attr', 'sinc', 'intel', 'fun', 'like',
    'prob', 'met', 'gender'
]]

dating_test = df_test[[
    'match', 'int_corr', 'samerace', 'age_o', 'race_o', 'pf_o_att', 'pf_o_sin',
    'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha', 'attr_o', 'sinc_o',
    'intel_o', 'fun_o', 'like_o', 'prob_o', 'met_o', 'age', 'race', 'imprace',
    'imprelig', 'goal', 'date', 'go_out', 'career_c', 'sports', 'tvsports',
    'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing',
    'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping',
    'yoga', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'attr3_1',
    'sinc3_1', 'fun3_1', 'intel3_1', 'attr', 'sinc', 'intel', 'fun', 'like',
    'prob', 'met', 'gender'
]]

# 针对训练集，删除有缺失值的行
print(date_df.shape)
date_df.dropna(inplace=True)
print(date_df.shape)

# 针对测试集，缺失位置补0
print(dating_test.shape)
dating_test = dating_test.fillna(0)
print(dating_test.shape)

# 数据整合
x = date_df.drop(columns=['match'])
y = date_df['match']

X_test = dating_test.drop(columns=['match'])
y_test = dating_test['match']

# 做训练集和测试集分割
X_train, X_valid, y_train, y_valid = train_test_split(x,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=0,
                                                      stratify=y)
print(X_train.shape, X_valid.shape)

print("采样前：{}".format(X_train.shape))

# 针对训练集，过采样
# oversample = imblearn.over_sampling.SVMSMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train)

# 下采样

# undersample=imblearn.under_sampling.RandomUnderSampler(random_state=34)
# X_train, y_train = undersample.fit_resample(X_train, y_train)

print("采样后：{}".format(X_train.shape))

# # 决策树
dtc = DecisionTreeClassifier(min_samples_leaf=10)

# 交叉验证
pre_valid_dtc = cross_val_score(dtc, X_train, y_train, cv=5)
print("决策树预测出的交叉验证准确率为：{}".format(pre_valid_dtc))
print(
    '------------------------------------------------------------------------------------'
)

dtc.fit(X_train, y_train)

# 验证
pre_valid = dtc.predict(X_valid)
print("决策树预测出的验证acc为：{} ， recall为：{} ， precision为：{} ， f1为：{}".format(
    metrics.accuracy_score(y_test, pre_valid),
    metrics.recall_score(y_test, pre_valid),
    metrics.precision_score(y_test, pre_valid),
    metrics.f1_score(y_test, pre_valid)))
print(
    '------------------------------------------------------------------------------------'
)

# 测试
pre_test = dtc.predict(X_test)

print("决策树预测出的测试acc为：{} ， recall为：{} ， precision为：{} ， f1为：{}".format(
    metrics.accuracy_score(y_test, pre_test),
    metrics.recall_score(y_test, pre_test),
    metrics.precision_score(y_test, pre_test),
    metrics.f1_score(y_test, pre_test)))
print(
    '------------------------------------------------------------------------------------'
)

# # 神经网络
mlp = MLPClassifier(random_state=2, max_iter=3000).fit(X_train, y_train)

# 交叉验证
pre_valid_mlp = cross_val_score(mlp, X_train, y_train, cv=5)
print("神经网络预测出的验证准确率为：{}".format(pre_valid_mlp))
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

# # 测试
pre_test_mlp = mlp.predict(X_test)
print("神经网络预测出的测试acc为：{} ， recall为：{} ， precision为：{} ， f1为：{}".format(
    metrics.accuracy_score(y_test, pre_test_mlp),
    metrics.recall_score(y_test, pre_test_mlp),
    metrics.precision_score(y_test, pre_test_mlp),
    metrics.f1_score(y_test, pre_test_mlp)))

from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense


def build_classifier():
    classifier = Sequential()
    classifier.add(
        Dense(units=4,
              kernel_initializer="uniform",
              activation="tanh",
              input_dim=x_sj.shape[1]))
    classifier.add(
        Dense(units=2, kernel_initializer="uniform", activation="tanh"))
    classifier.add(
        Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    classifier.compile(optimizer="adam",
                       loss="binary_crossentropy",
                       metrics=["accuracy"])
    return classifier


x_sj = (X_train.values.T).T
y_sj = (y_train.values.reshape(1, y_train.shape[0])).T

classifier = KerasClassifier(build_fn=build_classifier, epochs=20)
accuracies = cross_val_score(estimator=classifier, X=x_sj, y=y_sj, cv=3)
mean = accuracies.mean()

print(accuracies)
print("Accuracy mean :", mean)

classifier.fit(X_train, y_train)
preTest = classifier.predict(X_test)
print("神经网络预测出的测试acc为：{} ， recall为：{} ， precision为：{} ， f1为：{}".format(
    metrics.accuracy_score(y_test, preTest),
    metrics.recall_score(y_test, preTest),
    metrics.precision_score(y_test, preTest),
    metrics.f1_score(y_test, preTest)))
