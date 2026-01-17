"""
Metrics for 6DoF pose evaluation.
"""
import numpy as np


def calculate_eTE(gt_t, pr_t):
    return np.linalg.norm((pr_t-gt_t), ord=2)/10


def calculate_eRE(gt_R, pr_R):
    numerator = np.trace(np.matmul(gt_R, np.linalg.inv(pr_R))) - 1
    numerator = np.clip(numerator, -2, 2)
    return np.arccos(numerator/2)
