import numpy as np 
import torch


class q_exponential:
    def __init__(self, q, n = 20):
        self.q = q
        self.n = n
        self.pre_q = self.__calq__(q, n)
        
    def __calq__(self, q, n):
        pre_q = np.zeros(n+1, dtype=np.float32)
        pre_q[0] = 1
        
        for i in range(1, n+1):
            pre_q[i] = pre_q[i-1] * (1- q**i)
            
        return pre_q
    
    def q_exponential_series(self, x):
        if (self.q == 1):
            return torch.exp(x)

        # Initialize the series sum
        series_sum = 0.
        
        # Compute each term in the series up to the specified number of terms
        for n in range(0, self.n + 1):
            # Compute the numerator: (z^n) * (1-q)^n
            # numerator = (x ** n) * ((1 - self.q) ** n)
            
            # Add the term to the series sum
            series_sum += (((x ** n) * ((1-self.q)**n)) / self.pre_q[n])
            
           # print(f'n = {n}, the numerator is {x**n}, the denominator is {self.pre_q[n]}, res = {series_sum}...')
        
        return series_sum