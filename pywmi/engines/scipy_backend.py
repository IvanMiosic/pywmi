import logging
from scipy.optimize import minimize, LinearConstraint, Bounds, linprog
import numpy as np
import pypoman

from typing import List, Callable

from pywmi import Domain
from pywmi.smt_math import LinearInequality, Polynomial
from .convex_optimizer import ConvexOptimizationBackend
# import pysmt.shortcuts as smt

logger = logging.getLogger(__name__)


class ScipyOptimizer(ConvexOptimizationBackend):
    def __init__(self):
        super().__init__()
        self.method = 'trust-constr'
        # available methods: trust-constr, SLSQP

    @staticmethod
    def compute_gradient(domain: Domain, polynomial: Polynomial, sign: float = 1.0) -> Callable:
        return polynomial.compute_gradient_from_variables(sorted(domain.real_vars), sign)

    @staticmethod
    def compute_hessian(domain: Domain, polynomial: Polynomial, sign: float = 1.0) -> Callable:
        return polynomial.compute_hessian_from_variables(sorted(domain.real_vars), sign=sign)

    def optimize(self, domain: Domain, convex_bounds: List[LinearInequality],
                 polynomial: Polynomial, minimization: bool = True) -> dict:
        lower_bounds, upper_bounds = domain.get_ul_bounds()
        a, b = self.get_opt_matrices(domain, convex_bounds)

        if self.method == 'trust-constr':
            constraints = LinearConstraint(np.array(a), np.full(len(b), -np.inf), np.array(b))
        elif self.method == 'SLSQP':
            constraints = {'type': 'ineq',
                           'fun': lambda x: [b[i] - (np.dot(x, a[i])) for i in range(len(convex_bounds))],
                           'jac': lambda x: a}
        else:
            raise Exception("Unsupported method")

        bounds = Bounds(lower_bounds, upper_bounds)
        sign = 1.0 if minimization else -1.0
        obj_function = self.get_opt_function(domain, polynomial, sign)
        jacobian = self.compute_gradient(domain, polynomial, sign)
        hessian = self.compute_hessian(domain, polynomial, sign)
        initial_value = np.array(pypoman.polyhedron.compute_chebyshev_center(np.array(a), np.array(b)))

        result = minimize(obj_function, initial_value,
                          method=self.method, constraints=constraints,
                          jac=jacobian, hess=hessian,
                          options={'disp': True}, bounds=bounds)
        return {'value': sign * result.fun, 'point': dict(zip(sorted(domain.real_vars), result.x))}

    def __str__(self):
        return "scipy_opt"
