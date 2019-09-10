import logging
from scipy.optimize import minimize, LinearConstraint, Bounds, linprog
from scipy.sparse import coo_matrix
import numpy as np
import ipopt
import pypoman

from typing import List, Callable

from pywmi import Domain
from pywmi.smt_math import LinearInequality, Polynomial
from .convex_optimizer import ConvexOptimizationBackend
# import pysmt.shortcuts as smt

logger = logging.getLogger(__name__)


class IpoptOptimizer(ConvexOptimizationBackend):
    def __init__(self):
        super().__init__(True)

    class OptProblem(object):
        def __init__(self, domain, polynomial, a, sign):
            self.domain = domain
            self.polynomial = polynomial
            self.a = a
            self.hs = coo_matrix(np.tril(np.ones((len(domain.real_vars), len(domain.real_vars)))))
            self.compute_value = polynomial.get_function(sorted(self.domain.real_vars), sign)
            self.compute_gradient = polynomial.compute_gradient_from_variables(
                sorted(self.domain.real_vars), sign)
            self.compute_hessian = polynomial.compute_hessian_from_variables(
                sorted(self.domain.real_vars), hs=self.hs, sign=sign)

        def objective(self, x):
            return self.compute_value(x)

        def gradient(self, x):
            return self.compute_gradient(x)

        def constraints(self, x):
            return self.a @ x

        def jacobian(self, x):
            return np.concatenate(tuple([row for row in self.a]))

        def hessianstructure(self):
            return self.hs.col, self.hs.row

        def hessian(self, x, lagrange, obj_factor):
            return self.compute_hessian(x, lagrange, obj_factor)

    def optimize(self, domain, convex_bounds: List[LinearInequality],
                 polynomial: Polynomial, minimization: bool = True) -> dict or None:
        lower_bounds, upper_bounds = domain.get_ul_bounds()
        a, b = self.get_opt_matrices(domain, convex_bounds)
        initial_value = np.array(pypoman.polyhedron.compute_chebyshev_center(np.array(a), np.array(b)))
        sign = 1.0 if minimization else -1.0
        nlp = ipopt.problem(n=len(initial_value), m=len(convex_bounds) + 2*len(lower_bounds),
                            problem_obj=self.OptProblem(domain, polynomial, a, sign),
                            lb=lower_bounds, ub=upper_bounds,
                            cl=np.full(len(b), -np.inf), cu=b)
        nlp.addOption('mu_strategy', 'adaptive')
        nlp.addOption('print_level', 1)
        nlp.addOption('max_iter', 50)

        point, info = nlp.solve(initial_value)
        return {'value': sign*info['obj_val'],
                'point': dict(list(zip(sorted(domain.real_vars), point)))}

    def __str__(self):
        return "ipopt_opt"