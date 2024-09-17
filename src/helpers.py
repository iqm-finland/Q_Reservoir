
# -*- coding: utf-8 -*-
"""Collection of math functions to calculate error metrics and normalize data.
"""
import numpy as np
import scipy.stats as stats 
from sklearn.preprocessing import StandardScaler


def moving_average(x, w):
    """Moving average filter using convolution.
    
    Args:
        x (np.array): data of shape(steps, dimy)
        w (int): window size
    
    Returns:
        np.array: filtered data of shape(steps-w+1, dimy)
    """
    return np.convolve(x, np.ones(w), 'valid') / w

def correlation(pred, true):
    r"""Pearson correlation coefficient.

    Timesteps are rows, dims are colums.
    Implementation only works for dim=1?

    .. math::

        CC = \frac{cov^2 \left( s^{true}, s^{pred} \right)}{\sigma_{s^{true}}^2 \sigma_{s^{pred}}^2}

    Args:
        a (np.array): predictions of shape(steps, dimy)
        b (np.array): predictions of shape(steps, dimy)
    
    Returns:
        float in (-1, 1)
    """
    # 0: Non-working pearson
    # pearson = stats.pearsonr(x=pred, y=true) # doesnt work?
    # 1: Spearman
    # spearman = stats.spearmanr(a=pred, b=true, axis=0).statistic
    # 2: Squared Pearson Product-Moment Correlation Coefficient
    # https://arxiv.org/pdf/2302.03595.pdf
    # cov: offdiagonals are covariance (symmetric), diagonals are variance 
    squared_pearson = np.corrcoef(x=pred, y=true, rowvar=False)[0, 1]**2
    # 2.1: too many rounding errors
    # squared_pearson = (np.cov(m=pred, y=true, rowvar=False)[0, 1] \
    #     / (np.std(pred, axis=0) * np.std(true, axis=0)))[0]**2
    return squared_pearson

def mse(a, b):
    r"""Mean squared error.
    Scale dependent: don't use to compare different target sequences.

    .. math::

        \frac{1}{N_{t}} \sum_{t}^{N_t} \vert s_t^{pred} - s_t^{true} \vert^2
    
    Args:
        pred (np.array): predictions of shape(steps, dimy)
        true (np.array): predictions of shape(steps, dimy)
        
    Returns:
        float
    """
    if np.shape(a) != np.shape(b):
        raise Warning(f'''Shapes of arrays passed to mse() is not the same: 
            {np.shape(a)}, {np.shape(b)}.''')
    return ((a - b)**2).mean()

def mape(pred, true):
    r"""Mean Absolute Percentage Error (MAPE).
    Scale independent. Works best with data without zeros and extreme values

    .. math::

        \frac{100}{N_{t}} \sum_{t}^{N_t} \frac{\vert s_t^{pred} - s_t^{true} \vert}{s_t^{true}}
    
    Args:
        pred (np.array): predictions of shape(steps, dimy)
        true (np.array): predictions of shape(steps, dimy)
        
    Returns:
        float
    """
    return np.mean(np.abs(pred-true)/true) * 100

def nrmse(pred, true):
    r"""Normalized Root Mean Squared Error (NRMSE).

    .. math:: 

        \frac{ \left( 1/N \sum |y-y'|^2 \right)^{1/2} } { 1/N \sum y} \\

        \frac{ \sqrt{  \frac{1}{N_t} \sum_{t}^{N_t} \vert s_t^{pred} - s_t^{true} \vert^2 } } { \frac{1}{N_t} \sum_{t}^{N_t} s_t^{true}}

    Scale independent. Use for comparing different datasets or forecasting models.

    Args:
        pred (np.array): predictions of shape(steps, dimy)
        true (np.array): predictions of shape(steps, dimy)

    Returns:
        float
    """
    return np.sqrt( ((pred - true)**2).mean() ) / np.mean(true)

def weighted_gate_count(qc):
    """Number of single qubit gates + 10 * number of two qubit gates.
    
    Args:
        qc (QuantumCircuit)
    
    Returns:
        int
    """
    n = 0 
    for g, ng in qc.count_ops().items():
        if g in ['cx', 'cnot']:
            n += 10*ng
        else:
            n += ng
    return int(n)

def normalize(data, dmin, dmax): 
    """Normalization is a rescaling of the data from the original range so that 
    all values are within the range of 0 and 1.

    Works well if you have no outliers.
    new = (old - min) / (max - min)

    Args:
        data (list[np.array]): (episodes, steps, dims)
        dmin, dmax (list[int] | np.array[int]): current data range
    
    Returns:
        normalized_data (list[np.array]): (episodes, steps, dims)
    """
    # train the normalization on all episodes
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = scaler.fit(episode)
    # print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    dmin = np.asarray(dmin)
    dmax = np.asarray(dmax)
    normalized = []
    for episode in data:
        # normalize the episode
        normalized_episode = (episode - dmin) / (dmax - dmin)
        # normalized_episode = scaler.transform(episode)
        normalized.append(normalized_episode)
    return normalized 

def unnormalize(data, dmin, dmax):
    """Reverse scaling. (0, 1) -> (dmin, dmax).

    Args:
        data: list[np.array] (episodes, steps, dims)
        dmin, dmax: previous data range we want to return to
    
    Returns:
        unnormalized_data: list[np.array] (episodes, steps, dims)
    """
    dmin = np.asarray(dmin)
    dmax = np.asarray(dmax)
    oldmin = 0
    oldmax = 1
    normalized = []
    for episode in data:
        # normalize the episode
        normalized_episode = (episode - oldmin / (oldmax - oldmin)) * (dmax - dmin) + dmin
        normalized.append(normalized_episode)
    return normalized 

def unstandardize(data, dmin, dmax,doldmin,doldmax):
    """Reverse scaling. (doldmin, doldmax) -> (dmin, dmax).

    Args:
        data: list[np.array] (episodes, steps, dims)
        dmin, dmax: previous data range we want to return to
    
    Returns:
        unnormalized_data: list[np.array] (episodes, steps, dims)
    """
    dmin = np.asarray(dmin)
    dmax = np.asarray(dmax)
    oldmin = np.asarray(doldmin)
    oldmax = np.asarray(doldmax)
    normalized = []
    for episode in data:
        # normalize the episode
        normalized_episode = (episode - oldmin / (oldmax - oldmin)) * (dmax - dmin) + dmin
        normalized.append(normalized_episode)
    return normalized 

def standardize(data): 
    r"""Standardizing a dataset involves rescaling the distribution of values so 
    that the mean of observed values is 0 and the standard deviation is 1.

    Works well if data is approximately gaussian (normal).
    Skewed data distributions will remain skewed.

    .. math::
        x^{std}_t = (x_t - mean) / std \\
        mean = \sum_t x_t / N_t \\
        std = \sqrt{ 1/N_t \sum_t (x_t - mean)^2 }
    
    Args:
        data: list[np.array] (episodes, steps, dims)
    
    Returns:
        standardized_data: list[np.array] (episodes, steps, dims)
    """
    # train the standardization on all episodes
    scaler = StandardScaler()
    scaler = scaler.fit(np.vstack(data))
    print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, np.sqrt(scaler.var_)))
    # standardization the dataset and print the first 5 rows
    return [
        scaler.transform(episode)
        for episode in data
    ]