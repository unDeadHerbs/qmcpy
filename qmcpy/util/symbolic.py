"""Calculate the maximum and normalization constants for Acceptance Rejection."""

from contextlib import contextmanager
from qmcpy.util import ParameterError, NotYetImplemented
from scipy.stats import norm
from sympy.calculus.util import continuous_domain, singularities
import itertools
import numpy as np
import sympy as sp

# TODO: Type all of the functions


class SymbolicProxy:
    """A proxy object for wrapping sympy variables so that they can
    interact with comparison operators.

    """

    __array_priority__ = 1000

    def __init__(self, expression):
        self.expression = sp.sympify(expression)

    def _sympy_(self):
        return self.expression

    # NumPy calls ufuncs (sqrt, exp, sin, square, power, ...) through this
    # protocol.  Returning another proxy keeps symbolic expressions working
    # through ordinary NumPy-based user functions.
    _UFUNC_MAP = {
        "sqrt": sp.sqrt,
        "exp": sp.exp,
        "log": sp.log,
        "log2": lambda x: sp.log(x, 2),
        "log10": lambda x: sp.log(x, 10),
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "arcsin": sp.asin,
        "arccos": sp.acos,
        "arctan": sp.atan,
        "absolute": sp.Abs,
        "fabs": sp.Abs,
        "square": lambda x: x**2,
        "reciprocal": lambda x: 1 / x,
        "negative": lambda x: -x,
        "positive": lambda x: x,
        "power": lambda x, y: x**y,
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "true_divide": lambda x, y: x / y,
        "divide": lambda x, y: x / y,
        "maximum": sp.Max,
        "minimum": sp.Min,
    }

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        function = self._UFUNC_MAP.get(ufunc.__name__)
        if function is None:
            return NotImplemented
        args = [
            value.expression if isinstance(value, SymbolicProxy) else sp.sympify(value)
            for value in inputs
        ]
        result = function(*args)
        return SymbolicProxy(result)

    def __array_function__(self, func, types, args, kwargs):
        # Handle the common NumPy function forms that are not ufuncs.
        if func is np.sum:
            values = args[0]
            if isinstance(values, SymbolicProxy):
                return values
            return SymbolicProxy(sp.Add(*(sp.sympify(v) for v in values)))
        return NotImplemented

    def _binary(self, other, op):
        if isinstance(other, SymbolicProxy):
            return SymbolicProxy(op(self.expression, other._sympy_()))
        return SymbolicProxy(op(self.expression, sp.sympify(other)))

    def _condition(self, other, cond):
        return SymbolicProxy(
            sp.Piecewise((1, cond(self.expression, sp.sympify(other))), (0, True))
        )

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._binary(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._binary(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._binary(other, lambda a, b: b * a)

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._binary(other, lambda a, b: b / a)

    def __pow__(self, other):
        return self._binary(other, lambda a, b: a**b)

    def __rpow__(self, other):
        return self._binary(other, lambda a, b: b**a)

    def __neg__(self):
        return -self.expression

    def __pos__(self):
        return self.expression

    def __abs__(self):
        return sp.Abs(self.expression)

    def __lt__(self, other):
        return self._condition(other, sp.Lt)

    def __le__(self, other):
        return self._condition(other, sp.Le)

    def __gt__(self, other):
        return self._condition(other, sp.Gt)

    def __ge__(self, other):
        return self._condition(other, sp.Ge)

    def __eq__(self, other):
        return self._condition(other, sp.Eq)

    def __ne__(self, other):
        return self._condition(other, sp.Ne)

    def __bool__(self):
        raise NotYetImplemented(
            "Conditional statements (if/while) are not supported when converting "
            "a function to a symbolic expression."
        )


class SymbolicInput:
    def __init__(self, symbols):
        if isinstance(symbols, int):
            self.symbols = [sp.Symbol(f"x{i}", real=True) for i in range(symbols)]
        else:
            self.symbols = list(symbols)
        self.shape = (len(self.symbols), 1)

    def __getitem__(self, key):
        # Support the x[:, 0] convention.  Return a proxy so comparisons
        # are indicator functions.
        if isinstance(key, tuple):
            _, col = key
            if isinstance(col, int):
                return SymbolicProxy(self.symbols[col])
        elif isinstance(key, int):
            # TODO: Do I need this mode, or is that invalid for the
            # rest of qmcpy?
            return SymbolicProxy(self.symbols[key])
        raise NotImplementedError(f"Unsupported indexing: {key}")


@contextmanager
def scipy_to_sympy():
    """Monkeypatch out scipy functions temporarily."""
    # TODO: There has to be a better and safer to do this.  I
    # particularly don't like that I'm reimplementing complex
    # functions.
    mapping = {
        norm.pdf: (
            norm,
            lambda x, loc=0, scale=1: sp.exp(-(((x - loc) / scale) ** 2) / 2)
            / (scale * sp.sqrt(2 * sp.pi)),
        )
        # TODO: More functions from norm
    }
    original = {}
    try:
        for func, (package, sp_func) in mapping.items():
            name = func.__name__
            original[name] = (package, getattr(package, name))
            setattr(package, name, sp_func)
        yield
    finally:
        for name, (package, func) in original.items():
            setattr(package, name, func)


def to_symbolic(psi_function, dimensions):
    """Convert a python function (np[data,dimensions] -> number) to a sympy expression."""
    x = SymbolicInput(dimensions)
    with scipy_to_sympy():
        psi = psi_function(x)
    if isinstance(psi, SymbolicProxy):
        psi = psi._sympy_()
    if isinstance(psi, np.ndarray):
        psi = psi.tolist()
        if len(psi) == 1:
            psi = psi[0]
        else:
            # psi = sp.Matrix(psi)
            raise NotImplementedError(f"Unsupported Function Return Type: {type(psi)}")
    if isinstance(psi, (int, float, np.number)):
        psi = sp.sympify(psi)
    elif not isinstance(psi, (sp.Basic, sp.Expr)):
        raise NotImplementedError(f"Unsupported Function Return Type: {type(psi)}")

    # Combine indicator arithmetic into a single Piecewise expression where
    # possible, which is substantially easier for SymPy to differentiate.
    psi = sp.piecewise_fold(psi)
    return psi, x


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
        """Find all surface/boundary-conditions for the volume under consideration."""
        coordinate_values = []
        conditions = [
            condition
            for piecewise in sp.preorder_traversal(psi)
            if isinstance(piecewise, sp.Piecewise)
            for _, condition in piecewise.args
            if condition is not True
        ]
        # I don't really like this method.  I'd rather look at the set
        # of all boundaries (including those of the cube) and then
        # take the intersection of each subset of those boundaries,
        # and then use those subspace as the solution spaces; but,
        # that will probably require replacing the simple `subs` call
        # below with solving for some of the variables to substitute
        # them in and reduce the equation's order.
        for xi in x.symbols:
            values = [0, xi, 1]
            for condition in conditions:
                for relation in sp.preorder_traversal(condition):
                    if (
                        isinstance(relation, sp.core.relational.Relational)
                        and xi in relation.free_symbols
                    ):
                        try:
                            roots = sp.solve(sp.Eq(relation.lhs, relation.rhs), xi)
                        except (NotImplementedError, ValueError):
                            raise  # TODO: Find an example problem and clean up the error message.
                        for root in roots:
                            if not (root.free_symbols - {xi}):
                                values.append(root)
            coordinate_values.append(list(dict.fromkeys(values)))
        yield from itertools.product(*coordinate_values)

    def candidates():
        """Generate all candidate (point, value) pairs."""
        for xs in x_boundaries():
            free = [xi for xi in x.symbols if xi in xs]
            subs = {xi: v for xi, v in zip(x.symbols, xs) if xi != v}

            def raw_candidates():
                if not free:
                    # Corners have no free variables left and scipy
                    # doesn't appreciate the empty gradients lists.
                    return [{}]
                psi_projected = psi.subs(subs)
                gradients = [sp.diff(psi_projected, xi) for xi in free]
                try:
                    return sp.solve(gradients, free, dict=True)

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
                        raise  # TODO: Find an example of this problem and clean up the error message.
                        return []
                    stationary = [
                        point
                        for cpoint in stationary
                        for point in enumerate_conditional_point(cpoint)
                    ]
                    if not isinstance(stationary, (sp.FiniteSet, list)):
                        raise NotImplementedError(
                            "Stationary points of equation is not manageable."
                        )
                    return [dict(zip(free, point)) for point in stationary]

            for cp in raw_candidates():
                cand = {**cp, **subs}
                for xi in cand:
                    if cand[xi] == xi:
                        # If a variable is still free, just make it zero
                        cand[xi] = 0
                if not all(
                    (isinstance(xi, (int, float)) or (xi.is_real and xi.is_finite))
                    and 0 <= xi <= 1
                    for xi in cand.values()
                ):
                    continue
                value = psi.subs(cand)
                if value.is_real is False:
                    raise ParameterError(
                        f"Non-real maximum found in domain " f"(psi({cand}) = {value})."
                    )
                if value < 0:
                    raise ParameterError(
                        f"Density function must not be negative "
                        f"(psi({cand}) = {value})."
                    )

                point = tuple(
                    cand[symbol] if symbol in cand else 0  # If an axis is still free
                    for symbol in x.symbols
                )
                yield point, value

    return max(candidates(), key=lambda c: float(c[1]))


# TODO: Give this a better name
def max_norm(psi_function, dimensions):
    def has_exception_handler(func):
        # I'm not a fan of inspecting the function; but, I prefer it
        # over silently doing the wrong thing.
        import dis

        return bool(dis.Bytecode(func).exception_entries)

    if has_exception_handler(psi_function):
        raise ParameterError("Exceptions as control flow are not supported.")
    psi, x = to_symbolic(psi_function, dimensions)
    maximum = symbolic_to_maximum(psi, x)
    # TODO: Extract calculating area to own function.
    pre_norm_area = sp.integrate(psi, *[(symbol, 0, 1) for symbol in x.symbols])
    if pre_norm_area.is_real is None:
        pre_norm_area = pre_norm_area.evalf(30)
    if not pre_norm_area.is_real:
        # TODO: No test checks this path.
        raise ParameterError(
            f"Non-real integral found (int[psi,(x,0,1)^d]={pre_norm_area})."
        )
    if not pre_norm_area.is_finite:
        # TODO: Is this checked?
        raise ParameterError(
            f"Non-finite integral found (int[psi,(x,0,1)^d]={pre_norm_area})."
        )
    if 0 >= pre_norm_area:
        raise ParameterError(
            f"Function integral must be strictly positive (int[psi,(x,0,1)^d]={pre_norm_area})."
        )
    return maximum, pre_norm_area
