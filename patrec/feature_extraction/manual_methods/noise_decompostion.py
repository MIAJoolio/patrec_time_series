from typing import Dict, Optional, Union, Callable, Tuple
import time

import numpy as np
from scipy import stats

from ..old_fe_classes import Base_decomposer, Decomposition_result, FE_component_type
from patrec.generation.ts_noise_generators import *

class Noise_evaluator(Base_decomposer):
    """
    Evaluates noise characteristics against common noise distributions.
    
    Provides multiple evaluation metrics:
    1. Distribution fitting tests (KS, Anderson-Darling)
    2. Moment comparisons (variance, kurtosis)
    3. Tail behavior analysis
    4. Goodness-of-fit metrics
    
    Parameters:
        test_methods: List of tests to perform ('ks', 'anderson', 'moment', 'tail')
        significance_level: Threshold for statistical tests
    """
    
    def __init__(self, 
                 test_methods: Tuple[str] = ('ks', 'anderson', 'moment'),
                 significance_level: float = 0.05):
        self.test_methods = test_methods
        self.alpha = significance_level
        
    def decompose(self, noise_component: np.ndarray) -> Decomposition_result:
        start_time = time.time()
        
        # Ensure we're working with residuals (mean ~0)
        centered_noise = noise_component - np.mean(noise_component)
        std_dev = np.std(centered_noise)
        
        results = {}
        features = {}
        
        # Test against normal distribution
        if 'ks' in self.test_methods:
            ks_stat, p_val = stats.kstest(centered_noise/std_dev, 'norm')
            results['normal_ks_test'] = {
                'statistic': ks_stat,
                'p_value': p_val,
                'is_normal': p_val > self.alpha
            }
        
        if 'anderson' in self.test_methods:
            anderson_result = stats.anderson(centered_noise)
            critical = anderson_result.critical_values[2]  # 5% significance
            results['normal_anderson_test'] = {
                'statistic': anderson_result.statistic,
                'critical_value': critical,
                'is_normal': anderson_result.statistic < critical
            }
        
        if 'moment' in self.test_methods:
            features.update({
                'skewness': stats.skew(centered_noise),
                'kurtosis': stats.kurtosis(centered_noise, fisher=True),
                'variance': std_dev**2,
                'mean_abs_dev': np.mean(np.abs(centered_noise))
            })
        
        # Test against other distributions if significant deviations from normal
        if not results.get('normal_ks_test', {}).get('is_normal', True):
            if 'ks' in self.test_methods:
                # Test against exponential distribution
                exp_ks = stats.kstest(np.abs(centered_noise), 'expon')
                results['exponential_ks_test'] = {
                    'statistic': exp_ks.statistic,
                    'p_value': exp_ks.p_value
                }
                
                # Test against uniform distribution
                scaled = (centered_noise - np.min(centered_noise)) / \
                        (np.max(centered_noise) - np.min(centered_noise))
                uni_ks = stats.kstest(scaled, 'uniform')
                results['uniform_ks_test'] = {
                    'statistic': uni_ks.statistic,
                    'p_value': uni_ks.p_value
                }
        
        return Decomposition_result(
            component=centered_noise,
            component_type=FE_component_type.NOISE,
            method_name='noise_evaluator',
            params={
                'test_methods': self.test_methods,
                'significance_level': self.alpha
            },
            stats={
                'execution_time_sec': time.time() - start_time,
                'input_shape': noise_component.shape
            },
            features={
                'distribution_tests': results,
                'moment_analysis': features
            }
        )


class Synthetic_noise_generator:
    """
    Generates synthetic noise for testing/evaluation purposes.
    Wrapper around your existing noise generation functions with additional features.
    
    Parameters:
        noise_type: Type of noise ('normal', 'poisson', 'uniform', 'exponential')
        **params: Parameters specific to each noise type
    """
    
    def __init__(self, noise_type: str, **params):
        self.noise_type = noise_type
        self.params = params
        
    def generate(self, data: np.ndarray) -> np.ndarray:
        """Generate noise matching the input data's characteristics"""
        if self.noise_type == 'normal':
            return normal_noise(data, **self.params)
        elif self.noise_type == 'poisson':
            return poisson_noise(data, **self.params)
        elif self.noise_type == 'uniform':
            return uniform_noise(data, **self.params)
        elif self.noise_type == 'exponential':
            return exponential_noise(data, **self.params)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def evaluate_fit(self, target_noise: np.ndarray) -> Dict[str, float]:
        """
        Evaluate how well the generated noise matches target noise.
        Returns goodness-of-fit metrics.
        """
        generated = self.generate(np.zeros_like(target_noise))
        return {
            'correlation': np.corrcoef(generated, target_noise)[0,1],
            'mse': np.mean((generated - target_noise)**2),
            'ks_statistic': stats.ks_2samp(generated, target_noise).statistic
        }