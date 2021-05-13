from typing import Callable, Union, Iterable

import numpy as np
import pandas as pd
from pywt import wavedec
from scipy import stats, fftpack


def feature_extraction(input_path: str, output_path: str, train: bool = True):
    """特征提取"""
    data = pd.read_csv(input_path)

    data_processions = []

    def feat(*description: str, hidden: bool = False):
        def outer(f: Callable[[pd.Series, float], Union[float, Iterable[float]]]):
            data_processions.append((description, f, hidden))

        return outer

    @feat('最大值')
    def _(line, _):
        return line.max()

    @feat('最小值')
    def _(line, _):
        return line.min()

    @feat('均值')
    def _(line, _):
        return line.mean()

    @feat('方差')
    def _(line, _):
        return line.var()

    @feat('标准差')
    def _(line, _):
        return line.std()

    @feat('均方根')
    def _(line, _):
        return np.sqrt(np.square(line).mean())

    @feat('峰峰值')
    def _(_0, _1):
        return __['最大值'] - __['最小值']

    @feat('中位数')
    def _(line, _):
        return line.median()

    @feat('四分位差')
    def _(line, _):
        return np.percentile(line, 75) - np.percentile(line, 25)

    @feat('百分位差')
    def _(line, _):
        return np.percentile(line, 90) - np.percentile(line, 10)

    @feat('偏度', '峰度')
    def _(line, _):
        return *(getattr(stats, fn)(line) for fn in ('skew', 'kurtosis')),

    @feat('整流平均值')
    def _(line, _):
        return np.abs(line).mean()

    @feat('方根幅值')
    def _(line, _):
        return np.square(np.sqrt(np.abs(line)).mean())

    @feat('波形因子', '峰值因子', '脉冲值', '裕度')
    def _(*_):
        return *(__[_1] / __[_2] for _1, _2 in zip(['均方根'] + ['最大值'] * 3, ['整流平均值', '均方根', '整流平均值', '方根幅值'])),

    @feat('频域', '频率', hidden=True)
    def _(line, _):
        _1 = fftpack.fft(np.asarray(line))
        _2 = fftpack.fftfreq(len(line), 1 / 25600)
        _1 = abs(_1[_2 >= 0])
        _2 = _2[_2 >= 0]
        return _1, _2

    @feat(*('频域' + post for k, *_ in data_processions[:5] for post in k))
    def _(*_):
        _1 = __['频域']
        _2 = type(_1)
        return *(getattr(_2, fn)(_1) for fn in ['max', 'min', 'mean', 'var', 'std']),

    @feat('频域均方根')
    def _(*_):
        return np.sqrt(np.square(__['频域']).mean())

    @feat('频域中位数')
    def _(*_):
        return np.median(__['频域'])

    @feat('频域四分位差')
    def _(*_):
        return np.percentile(__['频域'], 75) - np.percentile(__['频域'], 25)

    @feat('频域百分位差')
    def _(*_):
        return np.percentile(__['频域'], 90) - np.percentile(__['频域'], 10)

    @feat(*(f'频域参数F{i}' for i in range(2, 9)))
    def _(*_):
        _1 = __['频域']
        _2 = __['频域均值']
        _3 = np.square(_1 - _2).sum() / (len(_1) - 1)
        _4 = __['频率']
        _5 = _1.sum()
        _6 = np.multiply(np.square(_4), _1).sum()
        _7 = np.multiply(pow(_4, 4), _1).sum()
        return _3, *(pow(_1 - _2, p).sum() / len(_1) / pow(_3, p / 2) for p in (3, 4)), \
               np.multiply(_4, _1).sum() / _5, \
               np.sqrt(_6) / _5, \
               np.sqrt(_7) / _6, \
               _6 / np.sqrt(_7 * _1.sum())

    @feat(*(f'5级小波变换参数{p}{k}' for k in ('', '比率') for p in ['cA5'] + [f'cD{i}' for i in reversed(range(1, 6))]))
    def _(line, _):
        _1 = wavedec(line, 'db10', level=5)
        _2 = [np.square(_v).sum() for _v in _1]
        _3 = sum(_2)
        _4 = [_v / _3 for _v in _2]
        return *_2, *_4

    @feat('label')
    def _(_, label):
        return label

    if not train:
        data_processions = data_processions[:-1]

    processed_data = pd.DataFrame(columns=[k for p, _, h in data_processions for k in p if not h])

    for index, row in data.iterrows():
        __ = {}
        result = []
        for desc, func, hidden in data_processions:
            ret = func(row[:-1], row[-1]) if train else func(row, None)
            if not isinstance(ret, tuple):
                ret = (ret,)
            if hidden:
                for k, v in zip(desc, ret):
                    __[k] = v
            else:
                for k, v in zip(desc, ret):
                    __[k] = v
                    result.append(v)
        processed_data.loc[index] = result

    processed_data.to_csv(output_path, index=False)


__all__ = ['feature_extraction']

if __name__ == '__main__':
    feature_extraction('../data/train01.csv', '../data/train01_fea.csv')
    feature_extraction('../data/train07.csv', '../data/train07_fea.csv')
