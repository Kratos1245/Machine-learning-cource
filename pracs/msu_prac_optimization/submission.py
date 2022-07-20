import numpy as np
from typing import Callable, Tuple, Union, List

class f1:
    def __call__(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return x**2

    def grad(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return 2*x

    def hess(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return 2



class f2:
    def __call__(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return np.sin(3*x**(3/2) + 2) + x**2

    def grad(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return 4.5*x**(1/2)*np.cos(3*x**(3/2) + 2) + 2*x

    def hess(self, x: float):
        """
        Args:
            x: float
        Returns:
            float
        """
        return 2 +(9/4)*x**(-1/2)*np.cos(3*x**(3/2)+2) - (81/4)*x*np.sin(3*x**(3/2)+2)

class f3:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            float
        """
        
        return (x[0]-3.3)**2/4 + (x[1]+1.7)**2/15

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            np.ndarray of shape (2,)
        """
        return np.array([(x[0]-3.3)/2, (2*(x[1]+1.7)/15)])

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2, 2)
        Returns:
            np.ndarray of shape (2, 2)
        """
        return np.array([[0.5, 0], [0, 2/15]])

class SquaredL2Norm:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            float
        """
        
        return float(x @ x.T)

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n,)
        """
        return 2*x

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (n,)
        Returns:
            np.ndarray of shape (n, n)
        """
        self.n = len(x)
        
        return 2*np.eye(self.n)

class Himmelblau:
    def __call__(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            float
        """
        x, y = x[0], x[1]
        return float((x**2+y-11)**2 + (x + y**2 - 7)**2)

    def grad(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2,)
        Returns:
            numpy array of shape (2,)
        """
        x, y = x[0], x[1]
        self.x0 = 2*(-7 + x + y**2 + 2*x*(-11 + x**2 + y))
        self.x1 = 2*(-11 + x**2 + y + 2*y*(-7 + x + y**2))
        return np.array([self.x0, self.x1])

    def hess(self, x: np.ndarray):
        """
        Args:
            x: numpy array of shape (2, 2)
        Returns:
            numpy array of shape (2, 2)
        """
        x, y = x[0], x[1]
        self.h = np.zeros((2,2))
        
        self.h[0][0] = 12*x**2+4*y-42
        self.h[0][1] = 4*(x + y)
        self.h[1][0] = 4*(x + y)
        self.h[1][1] = 4*x + 12*y**2-26
        
        return self.h

class Rosenbrok:
    def __call__(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            float
        """

        assert x.shape[0] >= 2, "x.shape[0] должен быть >= 2"
        
        n = len(x)
        _sum = 0 #Служебная сумма
        for i in range(n - 1):
            _sum += self.__func(x[i], x[i+1])
        return _sum

    def grad(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            numpy array of shape (n,)
        """

        len_x = x.shape[0]

        assert len_x >= 2, "x.shape[0] должен быть >= 2"
        
        n = int(len_x)
        grad = np.zeros(n)
        
        for i in range(n):
            if i == n-1:
                f = lambda x: 200*(x[i] - x[i-1]**2)
                grad[i] = f(x)
            elif i == 0:
                f = lambda x: - 400*x[i]*(x[i+1] - x[i]**2) - 2*(1-x[i])
                grad[i] = f(x)
            else:
                f = lambda x: 200*(x[i] - x[i-1]**2) - 400*x[i]*(x[i+1] - x[i]**2) - 2*(1-x[i])
                grad[i] = f(x)               

        return grad

    def hess(self, x: np.ndarray):

        """
        Args:
            x: numpy array of shape (n,) (n >= 2)
        Returns:
            numpy array of shape (n, n)
        """

        len_x = x.shape[0]

        assert len_x >= 2, "x.shape[0] должен быть >= 2"
        
        n = int(len_x)
        
        hess = np.zeros((n,n))
        for i in range(n):
            if i == n-1:
                hess[i][i] = 200
                hess[i][i-1] = -400*x[i-1]
            elif i == 0:
                hess[i][i] = 1200*x[i]**2 - 400*x[i+1] + 2
                hess[i][i+1] = -400*x[i]
            else:
                hess[i][i] = 1200*x[i]**2-400*x[i+1]+202
                hess[i][i-1] = -400*x[i-1]
                hess[i][i+1] = -400*x[i]
        return hess
    
    def __func(self
               , x0: 'Член с номером i'
               , x1: 'Член с номером i+1'
              ) -> float:
        return float(100*(x1 - x0**2)**2 + (1-x0)**2)
    
    
    

def minimize(
        func: Callable,
        x_init: np.ndarray,
        learning_rate: Callable = lambda x: 0.1,
        method: str = 'gd',
        max_iter: int = 10_000,
        stopping_criteria: str = 'function',
        tolerance: float = 1e-2,
) -> Tuple:
    """
    Args:
        func: функция, у которой необходимо найти минимум (объект класса, который только что написали)
            (у него должны быть методы: __call__, grad, hess)
        x_init: начальная точка
        learning_rate: коэффициент перед направлением спуска
        method:
            "gd" - Градиентный спуск
            "newtone" - Метод Ньютона
        max_iter: максимально возможное число итераций для алгоритма
        stopping_criteria: когда останавливать алгоритм
            'points' - остановка по норме разности точек на соседних итерациях
            'function' - остановка по норме разности значений функции на соседних итерациях
            'gradient' - остановка по норме градиента функции
        tolerance: c какой точностью искать решение (участвует в критерии остановки)
    Returns:
        x_opt: найденная точка локального минимума
        points_history_list: (list) список с историей точек
        functions_history_list: (list) список с историей значений функции
        grad_history_list: (list) список с исторей значений градиентов функции
    """

    assert max_iter > 0, 'max_iter должен быть > 0'
    assert method in ['gd', 'newtone'], 'method can be "gd" or "newtone"!'
    assert stopping_criteria in ['points', 'function', 'gradient'], \
        'stopping_criteria can be "points", "function" or "gradient"!'
    
    def is_stop_points(
        x_old: "Старая координата"
        , x_new: "Новая координата"
        , e: "Допустимая ошибка остановки"
    ) -> bool:
        return abs(x_new - x_old) <= e
    
    def is_stop_function(
        y_old: "Старое значение функции"
        , y_new: "Новое значение функции"
        , e: "Допустимая ошибка остановки"
    ) -> bool:
        return abs(y_new - y_old) <= e
    
    def is_stop_gradient(
        grad_new: "Новое значение градиента" 
        , e: "Допустимая ошибка остановки"
    ) -> bool:
        return abs(grad) <= e
    
    #Переменные входа
    func = func
    x_init = x_init
    a = learning_rate
    method = method
    max_iter = max_iter
    stopping_criteria = stopping_criteria
    e = tolerance
    
    #Переменные вывода
    x_opt = np.array([])
    points_history_list = list()
    functions_history_list = list()
    grad_history_list = list()
    
    if method == 'gd':
        x_new = x_init
        y_new = func(x_new)
        grad = func.grad(x_new)
        
        points_history_list.append(x_new)
        functions_history_list.append(y_new)
        grad_history_list.append(grad)
        for k in range(max_iter):
            x_old = np.array(x_new)
            y_old = np.array(y_new)
    
            grad = np.array(func.grad(x_old))
            x_new = x_old - a(k)*grad
            y_new = func(x_new)
            
            #Записываем истории
            points_history_list.append(x_new)
            functions_history_list.append(y_new)
            grad_history_list.append(grad)
            
            #Проверяем критерии остановки
            check_stop = False
            if stopping_criteria == 'points':
                check_stop = is_stop_points(x_old, x_new, e)
                check_stop = np.array(check_stop)
            elif stopping_criteria == 'function':
                check_stop = is_stop_function(y_old, y_new, e)
                check_stop = np.array(check_stop)
            elif stopping_criteria == 'gradient':
                check_stop = is_stop_gradient(grad, e)
                check_stop = np.array(check_stop)
            else:
                raise ValueError('Error stopping criteria')
            
            if check_stop.all() or (k == max_iter-1):
                x_opt = x_new
                break
                
        x_opt = points_history_list[-1]
    elif method == 'newtone':

        
        x_new = x_init
        y_new = func(x_new)
        grad = func.grad(x_new)
        
        
        points_history_list.append(x_new)
        functions_history_list.append(y_new)
        grad_history_list.append(grad)
        for k in range(max_iter):
            x_old = np.array(x_new)
            y_old = np.array(y_new)
            
            grad = np.array(func.grad(x_old))
            hess = np.array(func.hess(x_old))
            if hess.ndim >= 2:
                inv_hess = np.array(np.linalg.inv(hess))
                x_new = x_old - a(k)*np.dot(inv_hess, grad.T)
                y_new = func(x_new)
            else:
                inv_hess = 1/hess
                x_new = x_old - a(k)*inv_hess*grad
                y_new = func(x_new)
                
            #Записываем истории
            points_history_list.append(x_new)
            functions_history_list.append(y_new)
            grad_history_list.append(grad)
            
            #Проверяем критерии остановки
            check_stop = False
            if stopping_criteria == 'points':
                check_stop = is_stop_points(x_old, x_new, e)
                check_stop = np.array(check_stop)
            elif stopping_criteria == 'function':
                check_stop = is_stop_function(y_old, y_new, e)
                check_stop = np.array(check_stop)
            elif stopping_criteria == 'gradient':
                check_stop = is_stop_gradient(grad, e)
                check_stop = np.array(check_stop)
            else:
                raise ValueError('Error stopping criteria')
            
            if (check_stop.all()) or (k == max_iter-1):
                break
        x_opt = points_history_list[-1]
                
    else:
        raise ValueError('Error method')
    
    return x_opt, points_history_list, functions_history_list, grad_history_list