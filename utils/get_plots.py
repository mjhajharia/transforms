from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from tqdm import tqdm
import arviz as az
import sys
sys.path.insert(1, 'utils')
from ess import get_ess_plot
from rmse import get_rmse_plot
from sample import sample
import pickle
from utils import list_params, list_transforms, transforms_labels

get_ess_plot('density', 'simplex', 'DirichletSymmetric', 'x', 0)
get_ess_plot('cdf', 'simplex', 'DirichletSymmetric', 'x', 0)
get_rmse_plot('simplex', 'DirichletSymmetric')