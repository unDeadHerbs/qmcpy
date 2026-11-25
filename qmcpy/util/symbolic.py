"""Calculate the maximum and normalization constants for Acceptance Rejection."""

import itertools
import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain, singularities
from qmcpy.util import ParameterError, NotYetImplemented

# TODO: Type all of the functions

class SymbolicInput:
    def __init__(self, symbols):
        if isinstance(symbols, int):
            self.symbols = [
                sp.Symbol(f"x{i}",real=True)
                for i in range(symbols)
            ]
        else:
            self.symbols = list(symbols)
        self.shape=(len(self.symbols),1)

    def __getitem__(self, key):
        # Support x[:, 0] convention
        if isinstance(key, tuple):
            _, col = key
            if isinstance(col, int):
                return self.symbols[col]
        elif isinstance(key, int):
                return self.symbols[key]
        else:
            raise NotImplementedError(
                f"Unsupported indexing: {key}"
            )


import numpy as np
import sympy as sp
from scipy.stats import norm
from contextlib import contextmanager


@contextmanager
def numpy_to_sympy():
    """Monkeypatch out numpy and scipy functions temporarily"""

    # TODO: There has to be a better and safer to do this.  I
    # particularly don't like that I'm reimplementing scipy's norm
    # package.  Perhaps SymbolicInput can detect that it's being
    # handed to these functions and then dispatch correctly?
    mapping = {
        np.sqrt: (np,sp.sqrt),
        np.exp: (np,sp.exp),
        np.log: (np,sp.log),
        np.sin: (np,sp.sin),
        np.cos: (np,sp.cos),
        np.tan: (np,sp.tan),
        np.sinh: (np,sp.sinh),
        np.cosh: (np,sp.cosh),
        np.tanh: (np,sp.tanh),
        np.abs: (np,sp.Abs),
        np.absolute: (np,sp.Abs),
        np.square: (np,lambda z: z**2),
        np.power: (np,lambda z, p: z**p),
        norm.pdf: (norm, lambda x, loc=0, scale=1:
                   sp.exp(-((x-loc)/scale)**2/2)
                   / (scale*sp.sqrt(2*sp.pi)))
        # TODO: More functions from norm
    }
    original = {}
    try:
        for func, (package, sp_func) in mapping.items():
            name = func.__name__
            original[name] = (package,getattr(package, name))
            setattr(package, name, sp_func)
        yield
    finally:
        for name, (package, func) in original.items():
            setattr(package, name, func)

        
def to_symbolic(psi_function,dimensions):
    """Convert a python function (np[data,dimensions] -> number) to a sympy expression."""
    x = SymbolicInput(dimensions)
    with numpy_to_sympy():
        psi = psi_function(x)
    if isinstance(psi, np.ndarray):
        psi = psi.tolist()
        if len(psi) == 1:
            psi = psi[0]
        else:
            # psi = sp.Matrix(psi)
            raise NotImplementedError(
                f"Unsupported Function Return Type: {type(psi)}"
            )
    if isinstance(psi, (int, float, np.number)):
        psi = sp.sympify(psi)
    elif False: # TODO: not isinstance(psi,sp.something)
        raise NotImplementedError(
            f"Unsupported Function Return Type: {type(psi)}"
        )

    return psi,x


def enumerate_conditional_point(point):
    """The function `nonlinsolve` sometimes returns `ConditionSet`s
    instead of number like objects.  Splits those out into values.

    """
    coordinate_sets = []
    for coordinate in point:
        if isinstance(coordinate, sp.ConditionSet):
            solutions = sp.solveset(
                coordinate.condition,
                coordinate.sym,
                domain=sp.Interval(0, 1),
            )
        else:
            solutions = sp.FiniteSet(coordinate)
        if isinstance(solutions, sp.FiniteSet):
            coordinate_sets.append(list(solutions))
        elif solutions is sp.S.EmptySet:
            return []
        else:
            raise ValueError(f"Cannot explicitly enumerate solutions: {solutions}")
    return list(itertools.product(*coordinate_sets))


def symbolic_to_maximum(psi, x):
    """Find the maximum value that psi(x) takes over the unit interval."""
    def x_boundaries():
        """Find all surfaces/boundry-conditions of the voluem under consideration."""
        yield from itertools.product(*[(0, xi, 1) for xi in x.symbols])

    def candidates():
        """Generate all candidate (point, value) pairs."""
        for xs in x_boundaries():
            free = [xi for xi in x.symbols if xi in xs]
            subs = {
                xi: v
                for xi, v in zip(x.symbols, xs)
                if xi != v
            }
            def raw_candidates():
                if not free:
                    # Corners have no free variables left and scipy
                    # doesn't apprecate the empty gradients lists.
                    return [{}]
                psi_projected = psi.subs(subs)
                gradients = [sp.diff(psi_projected, xi) for xi in free]
                try:
                    return sp.solve(
                        gradients,
                        free,
                        dict=True
                    )
                    
                except:
                    # TODO: The function `nonlinsolve` can silently
                    # miss points.  Try partitioning the domain of a
                    # function like `abs` so that simpler functions
                    # can be formed.  Then `solve` can be used on the
                    # simplified functions and the solutions can be
                    # checked if they are in the correct domain for
                    # that simplification.  Note: I'd rather not
                    # support a type of function than provide an
                    # incorrect answer.
                    try:
                        stationary = sp.nonlinsolve(gradients, free)
                    except ValueError:
                        print(f"# TODO: This is probably an error, rather than masking?")
                        return []
                    stationary = [
                        point
                        for cpoint in stationary
                        for point in enumerate_conditional_point(cpoint)
                    ]
                    if not isinstance(stationary, (sp.FiniteSet, list)):
                        print(f"# TODO: Raise an error?")
                        return []
                    return [
                        dict(zip(free, point))
                        for point in stationary
                    ]

            for cp in raw_candidates():
                cand = {**cp, **subs}
                for xi in cand:
                    if cand[xi]==xi:
                        # If a variable is still free, just make it zero
                        cand[xi]=0
                if not all(
                        (isinstance(xi,(int,float))
                         or (xi.is_real and xi.is_finite))
                        and 0<=xi<=1
                        for xi in cand.values()
                ):
                    continue
                value = psi.subs(cand)
                if value.is_real is False:
                    raise ParameterError(
                        f"Non-real maximum found in domain "
                        f"(psi({cand}) = {value})."
                    )
                if value < 0:
                    raise ParameterError(
                        f"Density function must not be negative "
                        f"(psi({cand}) = {value})."
                    )

                point = tuple(
                    cand[symbol]
                    if symbol in cand else 0 # If an axis is still free
                    for symbol in x.symbols
                )
                yield point, value

    return max(
        candidates(),
        key=lambda c: float(c[1])
    )


def max_norm(psi_function,dimensions):
    psi, x = to_symbolic(psi_function,dimensions)
    maximum = symbolic_to_maximum(psi, x)
    # TODO: Extract calculating area to own function.
    pre_norm_area = sp.integrate(psi, *[(symbol, 0, 1) for symbol in x.symbols])
    if pre_norm_area.is_real is None:
        pre_norm_area = pre_norm_area.evalf(30)
    if not pre_norm_area.is_real:
        # TODO: No test checks this path.
        raise ParameterError(f"Non-real integral found (int[psi,(x,0,1)^d]={pre_norm_area}).")
    if not pre_norm_area.is_finite:
        # TODO: Is this checked?
        raise ParameterError(f"Non-finite integral found (int[psi,(x,0,1)^d]={pre_norm_area}).")
    if 0>=pre_norm_area:
        raise ParameterError(f"Function integral must be strictly positive (int[psi,(x,0,1)^d]={pre_norm_area}).")
    return maximum,pre_norm_area
