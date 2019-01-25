import pysmt.shortcuts as smt

from pywmi import Domain
from pywmi.engines.xsdd.smt_to_sdd import SddConversionWalker, convert, recover
from pywmi.smt_print import pretty_print

try:
    from pysdd.sdd import SddManager
except ImportError:
    pass


def test_convert_weight():
    converter = SddConversionWalker(SddManager())
    x, y = smt.Symbol("x", smt.REAL), smt.Symbol("y", smt.REAL)
    a = smt.Symbol("a", smt.BOOL)
    formula = smt.Ite((a & (x > 0) & (x < 10) & (x * 2 <= 20) & (y > 0) & (y < 10)) | (x > 0) & (x < y) & (y < 20), x + y, x * y) + 2
    result = converter.walk_smt(formula)
    print(result)
    print(converter.abstractions)
    print(converter.var_to_lit)


def test_convert_support():
    converter = SddConversionWalker(SddManager())
    x, y = smt.Symbol("x", smt.REAL), smt.Symbol("y", smt.REAL)
    a = smt.Symbol("a", smt.BOOL)
    formula = ((x <= 0) | (~a & (x <= -1))) | smt.Ite(a, x <= 4, x <= 8)
    print(pretty_print(formula))
    result = converter.walk_smt(formula)
    print(result)
    print(converter.abstractions)
    print(converter.var_to_lit)
    recovered = recover(result, converter.abstractions, converter.var_to_lit)
    print(pretty_print(recovered))
    with smt.Solver() as solver:
        solver.add_assertion(~smt.Iff(formula, recovered))
        if solver.solve():
            print(solver.get_model())
            assert False


def test_convert_weight2():
    domain = Domain.make(["a", "b"], ["x", "y"], [(0, 1), (0, 1)])
    a, b, x, y = domain.get_symbols(domain.variables)
    ite_a = smt.Ite(a, smt.Real(0.6), smt.Real(0.4))
    ite_b = smt.Ite(b, smt.Real(0.8), smt.Real(0.2))
    ite_x = smt.Ite(x >= smt.Real(0.5), smt.Real(0.5) * x + smt.Real(0.1) * y, smt.Real(0.1) * x + smt.Real(0.7) * y)
    weight = ite_a * ite_b * ite_x

    abstractions_c, var_to_lit_c = dict(), dict()
    converted_c = convert(smt.Real(0.6), SddManager(), abstractions_c, var_to_lit_c)
    for p, s in converted_c.items():
        print(f"{p}: {recover(s, abstractions_c, var_to_lit_c)}")
    assert len(converted_c) == 1

    abstractions_a, var_to_lit_a = dict(), dict()
    converted_a = convert(ite_a, SddManager(), abstractions_a, var_to_lit_a)
    for p, s in converted_a.items():
        print(f"{p}: {recover(s, abstractions_a, var_to_lit_a)}")
    assert len(converted_a) == 2

    converted_b = convert(ite_b, SddManager())
    assert len(converted_b) == 2

    print("X")
    abstractions_x, var_to_lit_x = dict(), dict()
    converted_x = convert(ite_x, SddManager(), abstractions_x, var_to_lit_x)
    for p, s in converted_x.items():
        print(f"{p}: {recover(s, abstractions_x, var_to_lit_x)}")
    assert len(converted_x) == 2

    converted = convert(weight, SddManager())
    assert len(converted) == 2 * 2 * 2