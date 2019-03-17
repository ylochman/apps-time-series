import os
from os.path import join as pjoin
import glob
import numpy as np
from numpy import hstack as stack
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15,10)
mpl.rcParams['image.cmap'] = 'inferno'

def show_series(train, test=None, test_df=None, x_freq=50):
    train = train.fillna(method='backfill')
    if isinstance(test, np.ndarray):
        assert test_df is not None
        test = pd.Series(test, index=test_df.index)
    plt.plot(train.index, train.values, label='train', color='C0')
    if test is not None:
        minv = stack((train.values, test.values)).min()
        maxv = stack((train.values, test.values)).max()
        plt.vlines(train.index[-1], min(0, minv), maxv, 'gray', '--')
        plt.plot(test.index, test.values, label='test', color='C1')
        if len(test) == 1:
            plt.scatter(test.index, test.values, color='C1')
        plt.xticks(stack((train.index, test.index))[::x_freq])
    else:
        minv = train.values.min()
        maxv = train.values.max()
#         try:
#             plt.xticks(pd.date_range(start=train.index[0], end=train.index[-1], freq='d').date.astype(str)[::x_freq])
#         except:
        plt.xticks(train.index[::x_freq])
#         plt.xticks(train.index.astype('datetime64[ns]').date.astype(str)[::x_freq])
    plt.legend()
    plt.yticks(stack(np.linspace(min(0, minv), maxv, 10)))
#     plt.show()

def split_ts(series, n_train):
    N = len(series)
    train = series.iloc[:n_train]
    test = series.iloc[n_train:]
    n_test = len(test)
    print('{}: {} + {}'.format(N, n_train, n_test))
    show_series(train, test, x_freq=40)
    return train, test, n_test


def divisors(x):
    divisors = []
    for i in range(1,x):
        if x%i == 0:
            divisors.append(i)
    return divisors

#######################################################################################################################
def trend_MA(train, k=3, method='simple'):
    N = len(train)
    if k % 2 == 1:
        assert method in ['simple', 'double']
        m = (k - 1) // 2 if method == 'simple' else (k + 1) // 2
    else:
        assert method in ['centered']
        m = k // 2
    trend = [np.nan] * m
    if method == 'simple':
        trend.extend(np.stack([train[k_:N-(k-(k_+1))] for k_ in range(k)]).mean(0))
    elif method == 'centered':
        trend.extend(np.stack([train[:N-k] * 0.5] +
                              [train[k_:N-(k-k_)] for k_ in range(1,k)] +
                              [train[k:N+1] * 0.5]).sum(0) / k)
    elif method == 'double':
        trend.extend(np.stack([train[:N-k-1] / 3] +
                              [train[1:N-k] * 2 / 3] +
                              [train[k_:N-(k+1-k_)] for k_ in range(2,k)] +
                              [train[k:N-1] * 2 / 3] +
                              [train[k+1:N] / 3]).sum(0) / k)
    trend.extend([np.nan] * m)
    return pd.Series(trend, index=train.index)

def MA_(train, k=10):
    n_train = len(train)
    forecast_values = []
    for i in range(k,n_train):
        forecast_values.append(train[i-k:i].mean())
    return pd.Series(forecast_values, index=train[k:].index)

def MA(train, test, k=10):
    n_test = len(test)
    forecast_values = []
    for i in range(n_test):
        train_k = stack((train.values[n_train-k+i:], forecast_values[max(i-k,0):i]))
        forecast_values.append(train_k.mean())
    return pd.Series(forecast_values, index=test.index)

def EWMA(train, test, alpha=0.5):
    n_train, n_test = len(train), len(test)
    forecast_values = [train.values[0]]
    for i in range(1,n_train):
        forecast_values.append(alpha * train.values[i-1] + (1 - alpha) * forecast_values[i-1])
    forecast_values.extend([forecast_values[-1]]*n_test)
    return pd.Series(forecast_values, index=stack((train.index, test.index)))

def Holt(train, test, alpha=0.3, beta=0.4):
    level, trend = [train.values[0]], [0]
    for i in range(1, n_train):
        level.append(alpha * train.values[i] + (1- alpha) * (level[i-1] + trend[i-1]))
        trend.append(beta * (level[i] - level[i-1]) + (1- beta) * trend[i-1])
#     forecast_values = level[-1] + trend[-1] * np.arange(n_test)
#     return pd.Series(forecast_values, index=test.index)
    forecast_values = stack((level, level[-1] + trend[-1] * np.arange(n_test)))
    return pd.Series(forecast_values, index=stack((train.index, test.index)))

def HoltWinters(train, test, alpha=0.3, beta=0.4, gamma=0.1, s=1):
    n_test = len(test)
    level, trend, season = [train.values[0]], [0], [0]
    for i in range(1, n_train):
        level.append(alpha * train.values[i] + (1- alpha) * (level[i-1] + trend[i-1]))
        trend.append(beta * (level[i] - level[i-1]) + (1- beta) * trend[i-1])
        season.append(gamma * (train.values[i] - level[i]) + (1 - gamma) * season[max(i-s, 0)])
#     forecast_values = level[-1] + trend[-1] * np.arange(n_test)
#     return pd.Series(forecast_values, index=test.index)
    S = level[-1] + trend[-1] * np.arange(n_test) + \
        stack((np.repeat([season[-s:]], n_test // s, 0).flatten(),
              season[:n_test % s]))
    forecast_values = stack((level + stack((np.zeros(s), season[:-s])), S))
    return pd.Series(forecast_values, index=stack((train.index, test.index)))


#######################################################################################################################
from statsmodels.tsa.stattools import acf

def ACF(y, y_pred=None, nlags=None, alpha=0.05, title=''):
    if y_pred is not None:
        y = y - y_pred
    nlags = len(y)//2 if nlags is None else nlags
    acfs, confs = acf(y, nlags=nlags, alpha=alpha)
#     plt.plot(confs[:,0], 'k--')
#     plt.plot(confs[:,1], 'k--')
    plt.stem(acfs, markerfmt='|')
    plt.title(title)
    plt.show()
    
# def acff(Y, h):
#     Y = np.array(Y)
#     N = len(Y)
#     M = Y.mean()
#     cov = ((Y[:N-h] - M)*(Y[h:] - M)).sum() / N
#     return cov / Y.var(ddof=0)

# def ACFF(Y, nlags=None):
#     if nlags is None:
#         nlags = len(Y)
#     acfs = np.array([acff(Y, h) for h in range(nlags)])
#     return acfs

# acfs = ACFF(train, 100)
# plt.stem(acfs, markerfmt='|')
# plt.show()
    
#######################################################################################################################
# Testing
#######################################################################################################################
from scipy import stats
from dm_test import dm_test

def normal_cdf(x):
    return stats.norm().cdf(x)

def normal_ppf(F):
    return stats.norm().ppf(F)

def two_sided_Z_test(x, a, sigma=None, alpha=None, out=True):
    '''
    H0: mean == a
    H1: mean != a
    alpha -- significance level
    '''
    if out:
        print('\nTwo-sided Z-test {}'.format('(sigma is unknown)' if sigma is None else ''))
        print('H0: mean == {}'.format(a))
        print('H1: mean != {}'.format(a))
    n = len(x)
    sample_mean = np.mean(x)
    if sigma is None:
        sample_var = np.var(x, ddof=1)
        v = (sample_mean - a) * np.sqrt(n) / np.sqrt(sample_var)
        p_value = 2 * student_cdf(n-1, v) if v < 0 else 2 * (1 - student_cdf(n-1, v))
        if alpha is not None:
            t_crit = student_ppf(n-1, 1-alpha/2)
    else:
        v = (sample_mean - a) * np.sqrt(n) / sigma
        p_value = 2 * normal_cdf(v) if v < 0 else 2 * (1 - normal_cdf(v))
        if alpha is not None:
            t_crit = normal_ppf(1-alpha/2)
            
    if out:
        print('p-value is {:.6f}. For all α > {:.6f} we can reject H0.'.format(p_value, p_value))
    if alpha is not None:
        rejected = v < -t_crit or v > t_crit
        if out:
            print('The rejection area is: (-inf, -{:.3f}) ∪ ({:.3f}, inf)'.format(t_crit, t_crit))
            print('t_stat = {:.3f} is {}in the rejection area.'.format(v, '' if rejected else 'not '))
            if not rejected:
                print('With significance level α={} we cannot reject H0.'.format(alpha, a))
            else:
                print('With significance level α={} we reject H0.'.format(alpha, a))
        return p_value, rejected

    return p_value

def one_sided_Z_test(x, a, sigma=None, alpha=None, left=False, out=True):
    '''
    H0: mean >= a
    H1: mean < a
    alpha -- significance level
    '''
    side = 'left' if left else 'right'
    if out:
        print('\nOne ({})-sided Z-test {}'.format(side, '(sigma is unknown)' if sigma is None else ''))
        print('H0: mean {}= {}'.format('>' if left else '<',a))
        print('H1: mean {} {}'.format('<' if left else '>',a))
    n = len(x)
    sample_mean = np.mean(x)
    if out:
        print('Sample mean: {:.3f}'.format(sample_mean))
    if sigma is None:
    # Unknown deviation
        sample_var = np.var(x, ddof=1)
        # Test statistic ~ St
        v = (sample_mean - a) * np.sqrt(n) / np.sqrt(sample_var)
        p_value = student_cdf(n-1, v) if left else (1 - student_cdf(n-1, v))
        if alpha is not None:
            t_crit = student_ppf(n-1, 1-alpha)
    else:
    # Known deviation
        # Test statistic ~ N(0, 1)
        v = (sample_mean - a) * np.sqrt(n) / sigma
        p_value = normal_cdf(v) if left else (1 - normal_cdf(v))
        if alpha is not None:
            t_crit = normal_ppf(1-alpha)
            
    if out:
        print('p-value is {:.2E}. For all α > {:.2E} we can reject H0.'.format(p_value, p_value))
    if alpha is not None:
        rejected = v < -t_crit if left else v > t_crit
        if out:
            reg_area = '(-inf, -{:.3f})'.format(t_crit) if left else '({:.3f}, inf)'.format(t_crit)
            print('The rejection area is: {}'.format(reg_area))
            print('t_stat = {:.3f} is {}in the rejection area.'.format(v, '' if rejected else 'not '))
            if not rejected:
                print('With significance level α={} we cannot reject H0.'.format(alpha, a))
            else:
                print('With significance level α={} we reject H0.'.format(alpha, a))
        return p_value, rejected
    return p_value


# from statsmodels.stats.descriptivestats import sign_test
# from scipy.stats import wilcoxon

loss_fn = lambda y, y_pred: (y - y_pred)**2

def EPA_sign(test, f1, f2, k1='first', k2='second', alpha=None): 
    d = loss_fn(test, f1) - loss_fn(test, f2)
    N = len(d)
    sample = (d > 0).astype(np.float)
    mean, std = 0.5, 0.5
    pvalue = one_sided_Z_test(sample, mean, std, alpha, left=False, out=False)
    stats.ttest_1samp(sample, mean)
    print('p-value is {:.2E}. For all α > {:.2E} we can reject H0.'.format(pvalue, pvalue))
    if pvalue < 0.05:
        print('Since the p-value is very small, we see that the {} model is better than {}.'.format(k2, k1))
        
def EPA_Wilcoxon(test, f1, f2, k1='first', k2='second', alpha=None): 
    d = loss_fn(test, f1) - loss_fn(test, f2)
    N = len(d)
    ranks = stats.rankdata(np.abs(d))
    sample = (d > 0).astype(np.float) * ranks
    mean = (N + 1) / 4
    std = np.sqrt((N+1) * (2 * N + 1) / 24)
    pvalue = one_sided_Z_test(sample, mean, std, alpha, left=False, out=False)
#     pvalue = wilcoxon(d).pvalue / 2 
    print('p-value is {:.2E}. For all α > {:.2E} we can reject H0.'.format(pvalue, pvalue))
    if pvalue < 0.05:
        print('The p-value is very small, so the {} model is much better than {}.'.format(k2, k1))
        
def EPA_Diebold_Mariano(test, f1, f2, k1='first', k2='second', alpha=None): 
    pvalue = dm_test(test, f1, f2).p_value
#     d = loss_fn(test, f1) - loss_fn(test, f2)
#     N = len(d)
#     M = np.ceil(N**(1/3))
#     d_mean = np.mean(d)
#     gamma = [np.sum((d - d_mean)[int(abs(i)):]*(d - d_mean)[:N-int(abs(i))])/N for i in np.arange(-M, M+1)]
#     std = np.sqrt(np.sum(gamma))
#     pvalue = one_sided_Z_test(d, 0, std, alpha, left=False, out=False)
    print('p-value is {:.2E}. For all α > {:.2E} we can reject H0.'.format(pvalue, pvalue))
    if pvalue < 0.05:
        print('The p-value is very small, so the {} model is better than {}.'.format(k2, k1))
        
        