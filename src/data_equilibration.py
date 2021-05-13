"""使测试数据均衡化，即在训练模型时使label=0～4均匀出现"""
from typing import Any

import pandas as pd
from imblearn.over_sampling import SMOTE

from utils import get_xy


def data_equilibration(input_path: str, output_path: str, random_state: Any = None):
    data = pd.read_csv(input_path)
    x, y = get_xy(data)

    smote = SMOTE(random_state=random_state)

    x, y = smote.fit_resample(x, y)

    res = pd.DataFrame(x)
    res['Label'] = y
    res.to_csv(output_path, index=False)


__all__ = ['data_equilibration']

if __name__ == '__main__':
    data_equilibration('../data/train01_fea.csv', '../data/train01_fea_smoted.csv')
    data_equilibration('../data/train07_fea.csv', '../data/train07_fea_smoted.csv')
