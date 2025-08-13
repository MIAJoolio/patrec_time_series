# seasonality and trend opt func
from typing import Literal
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import (pearsonr, spearmanr, kendalltau, 
                         pointbiserialr, weightedtau)
from scipy.spatial.distance import cosine
import numpy as np

def seasonal_correlation(result):
    """
    Estimate correlation between seasonal component and ideal signal
    Optimize: maximization
    Intervals: 
        good (0.3, 1]
        suspicious: [-0.3, 0.3]
        bad: (-0.3, -1]
        
    idea -> algo:Literal['dist', 'prototype']='dist', params:dict=None
    """
    # can be expanded with generators    
    # if params:
    
    t = result.get('t', np.arange(len(result['seasonal'])))
    seasonal_period = result.get('period', 12)
    seasonal_ampl = result.get('amplitude', 2)
    ideal_signal = np.sin(seasonal_ampl * np.pi * t / seasonal_period)

    corr, _ = pearsonr(result['seasonal'], ideal_signal)
    return corr


def seasonal_strength(result):
    """
    Estimate how greate changes in seasonal component and residuals
    Optimize: maximization
    Intervals: 
        good: [0.7, 1]
        sus: [0.3, 0.7)
        bad: [0, 0.3)
    """
    # if residuals.mean() >= 0.1:
    return max(0, 1 - np.var(result.get('resid')) / np.var(result.get('seasonal') + result.get('resid')))
    
    # return 0

def trend_strength(trend_component, residuals):
    """
    Estimate how greate changes in trend component and residuals
    Optimize: maximization
    Intervals: 
        good: [0.7, 1]
        sus: [0.3, 0.7)
        bad: [0, 0.3)
    """
    return max(0, 1 - np.var(residuals) / np.var(trend_component + residuals))


def find_period_range(series, strategy:Literal['mean', 'max', 'min']=None, ci_coef:int=2):
    
    if not strategy:
        peaks, _ = find_peaks(series, height=0)
        
        if len(peaks) >= 100:
            strategy = 'mean'
            print(f'applied max strategy, {len(peaks)=}')
        else:
            strategy = 'max'
            print(f'applied max strategy, {len(peaks)=}')
    
    if strategy == 'mean':
        peaks, _ = find_peaks(series, height=np.mean(series))
        return int(np.mean(peaks - np.concatenate([np.array([0]), peaks[0:]])[:-1]) * ci_coef)
    
    if strategy == 'min':
        peaks, _ = find_peaks(series, height=-np.abs(np.max(series)))
        return int(np.min(peaks - np.concatenate([np.array([0]), peaks[0:]])[:-1])) - ci_coef
    
    peaks, _ = find_peaks(series, height=np.median(series))

    return int(np.max(peaks - np.concatenate([np.array([0]), peaks[0:]])[:-1]) * ci_coef)