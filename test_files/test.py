import time

from pysmt.shortcuts import simplify
from pywmi import XsddEngine, Density, PredicateAbstractionEngine, XaddEngine, PyXaddEngine, PyXaddAlgebra, \
    RejectionEngine, XsddOptimizationEngine
from pywmi.domain import FileDensity
from pywmi.engines.latte_backend import LatteIntegrator
from pywmi.engines.scipy_backend import ScipyOptimizer
from pywmi.engines.ipopt_backend import IpoptOptimizer
from pywmi.smt_print import pretty_print

import logging
logging.basicConfig(level=logging.WARNING)


def main():
    # TODO Fix click graph

    # density = FileDensity.from_file("examples/uni_8")
    # density = FileDensity.from_file("examples/xor_6")
    # density = FileDensity.from_file("examples/mutex_6")
    density = FileDensity.from_file("examples/click_4")
    print("-----------------------------------------------------------")
    print("Support:")
    print(pretty_print(density.support))
    print()
    print("Weight:")
    print(pretty_print(density.weight))
    print()
    #print(density.support.to_smtlib())
    times = [time.time()]
    # result = NativeXsddEngine(density.domain, density.support, density.weight, LatteIntegrator(), factorized=False, find_conflicts=False, ordered=True).compute_volume()

    # PA
    # print("Result PA:", PredicateAbstractionEngine(density.domain, density.support, density.weight).compute_volume(add_bounds=False))
    # times.append(time.time())
    # print("Time PA: {:.4f}s".format(times[-1] - times[-2]))

    # XSDD:Latte
    # print("Result XSDD:", XsddEngine(density.domain, density.support, density.weight, LatteIntegrator(
    #    )).compute_volume(add_bounds=False))
    # times.append(time.time())
    # print("Time XSDD: {:.4f}s".format(times[-1] - times[-2]))

    # XSDD:PSI
    # print("Result XSDD(PSI):", XsddEngine(density.domain, density.support, density.weight).
    #       compute_volume(add_bounds=False))
    # times.append(time.time())
    # print("Time XSDD(PSI): {:.4f}s".format(times[-1] - times[-2]))
    
    # XSDD:OPT-ipopt
    result_opt = XsddOptimizationEngine(density.domain, density.support,
                                        density.weight, IpoptOptimizer()).\
        compute_optimum(add_bounds=False, minimization=False, exact=True)
    print("Result XSDD_OPT(ipopt):", result_opt['value'], "at", result_opt['point'])
    times.append(time.time())
    print("Time XSDD_OPT(ipopt): {:.4f}s".format(times[-1] - times[-2]))
    print()

    # XSDD:OPT-scipy
    # result_opt = XsddOptimizationEngine(density.domain, density.support,
    #                                     density.weight, ScipyOptimizer()). \
    #     compute_optimum(add_bounds=False, minimization=False, exact=True)
    # print("Result XSDD_OPT(scipy):", result_opt['value'], "at", result_opt['point'])
    # times.append(time.time())
    # print("Time XSDD_OPT(scipy): {:.4f}s".format(times[-1] - times[-2]))
    # print()

    # XSDD:BR
    #print("Result XSDD(BR):", XsddEngine(density.domain, density.support, density.weight, algebra=PyXaddAlgebra()).compute_volume(add_bounds=False))
    #times.append(time.time())
    #print("Time XSDD(BR): {:.4f}s".format(times[-1] - times[-2]))

    # print("Result REJ:", RejectionEngine(density.domain, density.support, density.weight, 1000000).compute_volume())
    # times.append(time.time())
    # print("Time REJ: {:.4f}s".format(times[-1] - times[-2]))

    # print("Result XSDD:F:", XsddEngine(density.domain, density.support, density.weight, None, factorized=True).compute_volume(add_bounds=False))
    # times.append(time.time())
    # print("Time XSDD:F: {:.4f}s".format(times[-1] - times[-2]))

    # print("Result XSDD:F:P:", XsddEngine(density.domain, density.support, density.weight, None, factorized=True).compute_volume(add_bounds=False))
    # times.append(time.time())
    # print("Time XSDD:F:P: {:.4f}s".format(times[-1] - times[-2]))

    # XSDD:F + XADD#
    # print("Result XSDD:FXADD:", XsddEngine(density.domain, density.support, density.weight, None, factorized=True, ordered=True, algebra=PyXaddAlgebra(reduce_strategy=(True, (True, True, True)))).compute_volume(add_bounds=False))
    # times.append(time.time())
    # print("Time XSDD:FXADD: {:.4f}s".format(times[-1] - times[-2]))

    # XADD
    # print("Result XADD:", XaddEngine(density.domain, density.support, density.weight).compute_volume(add_bounds=False))
    # times.append(time.time())
    # print("Time XADD: {:.4f}s".format(times[-1] - times[-2]))

    # result = NativeXsddEngine(density.domain, density.support, density.weight, LatteIntegrator(), factorized=False, find_conflicts=True, ordered=False).compute_volume(add_bounds=False)

    #print("Result PyXADD:", PyXaddEngine(density.domain, density.support, density.weight).compute_volume(add_bounds=False))
    #times.append(time.time())
    #print("Time PyXADD: {:.4f}s".format(times[-1] - times[-2]))

    # print(pretty_print(simplify(density.support)))
    
    #opt
    #print("Result Opt: ", PyXsddEngine(density.domain, density.support, density.weight).compute_optimum(add_bounds=False))
    #times.append(time.time())
    #print("Time Opt: {:.4f}s".format(times[-1] - times[-2]))


if __name__ == '__main__':
    main()