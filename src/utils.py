from pandas import DataFrame


def get_xy(data: DataFrame):
    """从DataFrame中获取数据(X)和标签(Y)，如果数据不包含标签，则为None"""
    if 'label' in data.columns:
        return data.drop(columns='label'), data.loc[:, 'label']
    elif 'Label' in data.columns:
        return data.drop(columns='Label'), data.loc[:, 'Label']
    else:
        return data, None


def classification_result_to_values(data):
    """将分类模型得到的结果转换为list[int]格式"""
    return [list(y).index(max(y)) for y in data]


def regression_result_to_values(data):
    """将回归模型得到的结果转换为list[int]格式"""
    return [round(y) for y in data]


__all__ = ['get_xy', 'classification_result_to_values', 'regression_result_to_values']
