from typing import List, Callable
from scipy.optimize import linprog
import numpy as np
import pypoman

from pywmi import Domain
from pywmi.smt_math import LinearInequality, Polynomial

class ConvexOptimizationBackend(object):
    def __init__(self, exact=True):
        self.exact = exact

    @staticmethod
    def get_opt_function(domain: Domain, polynomial: Polynomial, sign: float = 1.0) -> Callable:
        return polynomial.get_function(sorted(domain.real_vars), sign)

    @staticmethod
    def get_opt_matrices(domain: Domain, convex_bounds: List[LinearInequality]) -> (List, List):
        a = [[bound.a(var) for var in sorted(domain.real_vars)] for bound in convex_bounds]
        b = [bound.b() for bound in convex_bounds]
        # domain bounds
        lower_bounds, upper_bounds = domain.get_ul_bounds()
        for i in range(len(lower_bounds)):
            x = [0] * len(lower_bounds)
            x[i] = -1.0;
            a.append(x.copy())
            x[i] = 1.0;
            a.append(x.copy())
            b.append(-lower_bounds[i])
            b.append(upper_bounds[i])
        return a, b

    def is_nonempty(self, domain: Domain, convex_bounds: List[LinearInequality]) -> bool:
        a, b = self.get_opt_matrices(domain, convex_bounds)
        if linprog([0] * len(domain.real_vars), a, b).success:
            return True
        return False

    def Lipschitz_bound(self, domain, convex_bounds: List[LinearInequality],
                    polynomial: Polynomial, minimization: bool = True) -> float:
        """returns best possible optimum, computed using Lipschitz constant"""
        a, b = self.get_opt_matrices(domain, convex_bounds)
        polytope_vertices = np.array(pypoman.duality.compute_polytope_vertices(np.array(a), np.array(b)))
        sign = 1.0 if minimization else -1.0
        obj_function = self.get_opt_function(domain, polynomial, sign)
        point = np.full(len(domain.real_vars), 1/len(domain.real_vars))

        distance = np.max([np.linalg.norm(point - vertex) for vertex in polytope_vertices])
        max_values = np.maximum.reduce([np.absolute(vertex) for vertex in polytope_vertices])
        best_possible_opt = sign*obj_function(point)\
                            - sign * polynomial.max_jacobian(sorted(domain.real_vars), max_values) * distance
        return best_possible_opt

    def approximate_opt_by_sample(self, domain: Domain, convex_bounds: List[LinearInequality],
                                  polynomial: Polynomial, minimization: bool = True,
                                  sample_size: int = 10) -> float:
        a, b = self.get_opt_matrices(domain, convex_bounds)
        polytope_vertices = np.array(pypoman.duality.compute_polytope_vertices(np.array(a), np.array(b)))
        sign = 1.0 if minimization else -1.0
        obj_function = self.get_opt_function(domain, polynomial, sign)
        optimum = None
        while sample_size > 0:
            sample_size -= 1
            convex_coefficients = np.random.randint(0, len(polytope_vertices),
                                                    size=len(polytope_vertices))
            convex_coefficients = convex_coefficients / np.sum(convex_coefficients)
            point = np.dot(convex_coefficients, polytope_vertices)
            function_value = sign*obj_function(point)
            if optimum is None or sign*(function_value - optimum) < 0:
                optimum = function_value
        return optimum

    def optimize(self, domain: Domain, convex_bounds: List[LinearInequality],
                 polynomial: Polynomial, minimization: bool = True) -> dict or None:
        raise NotImplementedError()
