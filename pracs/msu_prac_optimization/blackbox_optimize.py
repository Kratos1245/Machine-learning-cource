import numpy as np
from typing import Tuple, Union
import numpy as np
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from matplotlib import pyplot
from typing import Tuple, Union
from sklearn.gaussian_process import GaussianProcessRegressor


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def sample():
    t = np.linspace(-20, 20, 400)
    Xs = cartesian_product(t,t) #нужно добавить 10 векторов
    return Xs

def acquisition(X, Y, model):
    
    X=X
    Y=Y
    model = model
    
    x_s = sample()
    mu, std = model.predict(x_s, return_std = True)
    
    best = max(Y)
    
    ix = argmax(norm.cdf(mu-best-1e-6/std))
    x_new = x_s[ix]
    
    return x_new

def get_y(origin): #Минус для минимизации
    return -origin(x_new)

def fit_new_model(X, Y, model):
    model = GaussianProcessRegressor()
    model.fit(X, Y)
    return model

def init(args_history):
    # Пример такой функции:
    if len(args_history) == 0:
        return np.array([0]*10)
    else:
        pass
    
    

def blackbox_optimize(
        args_history: np.ndarray,
        func_vals_history: np.ndarray
) -> Union[np.ndarray, str]:

    """
    Функция, которая по истории проверенных точек и значений blackbox функции в них возращает точку, которую следует
    проверить следующей или же строку "stop". Учтите случай, что должна выдавать функция, когда истории нет
    (args_history и func_vals_history это пустые arrays)

    Args:
        args_history: история аргументов (args_history.shape = (n, 10))
        func_vals_history: история значений функции в соответствующих аргументах
    Returns:
        Следующая точка (np.ndarray размера 10)
    """
    init(args_history)
    
    X = args_history
    Y = func_vals_history
    print(args_history)
    
    model = fit_new_model(X, Y)
    x_new = acquisition(X, Y, model)
    
    return x_new
    
    