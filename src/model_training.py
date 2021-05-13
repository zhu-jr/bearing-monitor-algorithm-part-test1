"""模型训练"""
from collections import Counter
from random import shuffle

import pandas as pd
import lightgbm as lgb

from src.feature_extraction import feature_extraction
from utils import get_xy, classification_result_to_values, regression_result_to_values
from sklearn.preprocessing import StandardScaler
from pickle import dump

# 下面这些模型仅用于打伪标签
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

ORIGIN_HUGE_TRAIN_DATA = pd.concat(
    (pd.read_csv('../data/train01_fea_smoted.csv'), pd.read_csv('../data/train07_fea_smoted.csv')),
    ignore_index=True)
ORIGIN_HUGE_TRAIN_DATA_X, ORIGIN_HUGE_TRAIN_DATA_Y = get_xy(ORIGIN_HUGE_TRAIN_DATA)

PUBLIC_SCALER = StandardScaler()
ORIGIN_HUGE_TRAIN_DATA_X = PUBLIC_SCALER.fit_transform(ORIGIN_HUGE_TRAIN_DATA_X)

with open('x_scaler.pkl', 'wb') as f:
    dump(PUBLIC_SCALER, f)
    
TEST_DATA = pd.read_csv('../data/test01_fea.csv')
TEST_DATA_X, _ = get_xy(TEST_DATA)
TEST_DATA_X = PUBLIC_SCALER.transform(TEST_DATA_X)


def make_pseudo_labels():
    models = (LinearRegression(),
              LogisticRegression(),
              RandomForestClassifier(),
              ExtraTreesClassifier(),
              DecisionTreeClassifier(),
              SVC(decision_function_shape='ovo'))
    # 存储几个模型的伪标签预测结果
    pseudo_labels = [[] for _ in range(TEST_DATA_X.shape[0])]
    for model in models:
        model.fit(ORIGIN_HUGE_TRAIN_DATA_X, ORIGIN_HUGE_TRAIN_DATA_Y)
        y_result = model.predict(TEST_DATA_X)
        try:
            # 假设这个模型是个回归模型
            y_result = regression_result_to_values(y_result)
        except TypeError:
            # 否则它是个分类模型
            y_result = classification_result_to_values(y_result)
        # 存储预测结果
        for arr, value in zip(pseudo_labels, y_result):
            arr.append(value)
    # 选取相同最多的那个标签作为真正的伪标签
    pseudo_labels = [max(Counter(lbls).items(), key=lambda v: v[1])[0] for lbls in pseudo_labels]
    # 再手动随机均衡化一次
    predicted_labels_counter = Counter(pseudo_labels)
    least_label_count = min(predicted_labels_counter.values())

    # 随机选取标签
    indexes = [[] for _ in range(5)]
    for index, label in enumerate(pseudo_labels):
        indexes[label].append(index)

    # 准备写入文件
    result = pd.DataFrame(columns=TEST_DATA.columns)
    result['label'] = ''  # 加入标签列

    data_frame_index_counter = 0

    for label, indexs in enumerate(indexes):
        shuffle(indexs)
        for index in indexs[:least_label_count]:
            result.loc[data_frame_index_counter] = list(TEST_DATA.loc[index].values) + [label]  # 未归一化

    result.to_csv('../data/test01_with_pseudo_label.csv', index=False)


def lightgbm_model(x, y):
    """lgb模型，主要使用的分类模型"""
    data_train = lgb.Dataset(x, label=y)
    clf = lgb.train({
        'learning_rate': 0.009,
        'objective': 'multiclass',
        'num_class': 5,

    }, train_set=data_train, valid_sets=[data_train])
    clf.save_model('model')

    def predictor(test_data):
        return clf.predict(test_data)

    return predictor


if __name__ == '__main__':
    make_pseudo_labels()
    pseudo_train_data = pd.read_csv('../data/test01_with_pseudo_label.csv')
    train = pd.concat([ORIGIN_HUGE_TRAIN_DATA, pseudo_train_data], ignore_index=True)
    train_x, train_y = get_xy(train)
    train_x = PUBLIC_SCALER.transform(train_x)
    p = lightgbm_model(train_x, train_y)
    feature_extraction('../data/test02.csv', '../data/test02_fea.csv')
    test = pd.read_csv('../data/test02_fea.csv')
    test_x, _ = get_xy(test)
    test_x = PUBLIC_SCALER.transform(test_x)
    predict = classification_result_to_values(p(test_x))

    predict_dataframe = pd.DataFrame()
    predict_dataframe['label'] = predict
    predict_dataframe.to_csv('../data/predict_result.csv', index=False)

