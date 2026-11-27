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

    def __init__(self, expression, shape=None):
        self.expression = sp.sympify(expression)
        # A proxy represents a symbolic value carried through NumPy's
        # array-oriented APIs. Infer the shape when the expression itself is
        # a SymPy N-dimensional array; scalar expressions represent one item.
        if shape is None:
            shape = (
                self.expression.shape
                if isinstance(self.expression, sp.NDimArray)
                else (1,)
            )
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape)) if self.shape else 1
        self.dtype = np.dtype(float)

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
        "ndtr": lambda x: (1 + sp.erf(x / sp.sqrt(2))) / 2,
    }

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented

        # In this workflow, isnan is only used by SciPy for error checking.
        # Symbolic values are treated as not-NaN; return a concrete NumPy
        # boolean mask with the proxy's recorded shape.
        if ufunc.__name__ == "isnan":
            return np.zeros(self.shape, dtype=np.bool_)

        # NumPy's bitwise/logical ufuncs are used by callers such as
        # scipy.stats to combine support-condition masks. Route them through
        # the same elementwise symbolic Boolean handling as Python's & and |.
        if ufunc.__name__ in ("bitwise_and", "logical_and"):
            other = next(value for value in inputs if value is not self)
            return self._logical(other, sp.And)
        if ufunc.__name__ in ("bitwise_or", "logical_or"):
            other = next(value for value in inputs if value is not self)
            return self._logical(other, sp.Or)
        if ufunc.__name__ in ("invert", "logical_not"):
            return self.__invert__()

        function = self._UFUNC_MAP.get(ufunc.__name__)
        if function is None:
            return NotImplemented
        args = [
            value.expression if isinstance(value, SymbolicProxy) else sp.sympify(value)
            for value in inputs
        ]
        # Apply ufuncs elementwise whenever any operand is a SymPy array.
        # NumPy may dispatch a binary ufunc with a scalar on the left and an
        # NDimArray on the right (for example, np.add(0, proxy.expression));
        # passing that combination directly to a scalar SymPy function fails.
        array_args = [arg for arg in args if isinstance(arg, sp.NDimArray)]
        if array_args:
            shape = array_args[0].shape
            if any(array.shape != shape for array in array_args[1:]):
                raise ValueError(
                    "Symbolic array operands must have matching shapes; "
                    f"got {[array.shape for array in array_args]}."
                )
            indices = itertools.product(*(range(length) for length in shape))
            values = []
            for index in indices:
                element_args = [
                    arg[index] if isinstance(arg, sp.NDimArray) else arg
                    for arg in args
                ]
                values.append(function(*element_args))
            result = sp.ImmutableDenseNDimArray(values, shape)
        else:
            result = function(*args)
            shape = self.shape
        return SymbolicProxy(result, shape=shape)

    def __array_function__(self, func, types, args, kwargs):
        # Handle the common NumPy function forms that are not ufuncs.
        # scipy.stats uses np.shape(condition) to allocate its output array;
        # returning the proxy's recorded shape avoids coercing a symbolic
        # condition into a concrete NumPy array.
        if func is np.shape:
            value = args[0]
            if isinstance(value, SymbolicProxy):
                return value.shape
        if func is np.ndim:
            value = args[0]
            if isinstance(value, SymbolicProxy):
                return value.ndim
        if func is np.size:
            value = args[0]
            if isinstance(value, SymbolicProxy):
                if len(args) > 1 or "axis" in kwargs:
                    axis = args[1] if len(args) > 1 else kwargs["axis"]
                    return value.shape[axis]
                return value.size
        if func is np.any:
            # SciPy uses np.any(cond) to decide whether to evaluate a
            # distribution's implementation. Reduce symbolic indicator
            # expressions when their truth value is determined by SymPy.
            value = args[0]
            if isinstance(value, SymbolicProxy):
                expression = value.expression
                if isinstance(expression, sp.NDimArray):
                    entries = [
                        expression[index]
                        for index in itertools.product(
                            *(range(length) for length in expression.shape)
                        )
                    ]
                else:
                    entries = [expression]
                conditions = []
                for entry in entries:
                    if entry in (sp.S.true, sp.S.false):
                        conditions.append(entry)
                    elif entry in (sp.S.Zero, sp.S.One):
                        conditions.append(sp.Ne(entry, 0))
                    else:
                        conditions.append(sp.Ne(entry, 0))
                result = sp.simplify(sp.Or(*conditions))
                if result is sp.S.true:
                    return True
                if result is sp.S.false:
                    return False
                raise NotYetImplemented(
                    "Cannot determine np.any() for an undecidable symbolic mask."
                )
        if func is np.all:
            # scipy.stats uses np.all(cond) to determine whether every
            # support-condition entry is true. Reduce the symbolic mask to a
            # native boolean when SymPy can determine the result.
            value = args[0]
            if isinstance(value, SymbolicProxy):
                if kwargs.get("axis", None) is not None or len(args) > 1:
                    raise NotYetImplemented(
                        "np.all() with an axis is not implemented for SymbolicProxy."
                    )
                expression = value.expression
                if isinstance(expression, sp.NDimArray):
                    entries = [
                        expression[index]
                        for index in itertools.product(
                            *(range(length) for length in expression.shape)
                        )
                    ]
                else:
                    entries = [expression]
                conditions = [
                    entry if entry in (sp.S.true, sp.S.false)
                    else sp.Ne(entry, 0)
                    for entry in entries
                ]
                result = sp.simplify(sp.And(*conditions))
                if result is sp.S.true:
                    return True
                if result is sp.S.false:
                    return False
                raise NotYetImplemented(
                    "Cannot determine np.all() for an undecidable symbolic mask."
                )
        if func is np.place:
            # scipy.stats uses np.place to fill the out-of-support portions
            # of its result. Resolve symbolic masks when they are definite;
            # an unresolved symbolic mask cannot safely index a concrete
            # NumPy output array.
            array, mask, values = args
            if isinstance(mask, SymbolicProxy):
                expression = mask.expression
                if isinstance(expression, sp.NDimArray):
                    entries = [expression[index] for index in itertools.product(
                        *(range(length) for length in expression.shape)
                    )]
                else:
                    entries = [expression]
                concrete_mask = []
                for entry in entries:
                    simplified = sp.simplify(entry)
                    if simplified in (sp.S.true, sp.S.One):
                        concrete_mask.append(True)
                    elif simplified in (sp.S.false, sp.S.Zero):
                        concrete_mask.append(False)
                    else:
                        raise NotYetImplemented(
                            "A symbolic mask with undecidable entries cannot "
                            "be used to update a concrete NumPy array."
                        )
                if isinstance(values, SymbolicProxy):
                    values = values.expression
                    
                    if isinstance(values, sp.NDimArray):
                        values = np.array(values.tolist(), dtype=object)
                    else:
                        values = np.asarray(values, dtype=object)

                np.place(array, np.asarray(concrete_mask, dtype=np.bool_).reshape(mask.shape), values)
                return None
        
        if func is np.broadcast_arrays:
            arrays = args
            shapes = [
                value.shape if isinstance(value, SymbolicProxy)
                else np.shape(value)
                for value in arrays
            ]
            broadcast_shape = np.broadcast_shapes(*shapes)
            results = []

            for value, shape in zip(arrays, shapes):
                if isinstance(value, SymbolicProxy):
                    expression = value.expression

                    if isinstance(expression, sp.NDimArray):
                        ndim_diff = len(broadcast_shape) - len(shape)
                        indices = itertools.product(
                            *(range(length) for length in broadcast_shape)
                        )
                        values = []
                        for index in indices:
                            source_index = tuple(
                                0 if length == 1 else index[ndim_diff + axis]
                                for axis, length in enumerate(shape)
                            )
                            values.append(expression[source_index])

                        expression = sp.ImmutableDenseNDimArray(
                            values, broadcast_shape
                        )

                    results.append(
                        SymbolicProxy(expression, shape=broadcast_shape)
                    )
                else:
                    results.append(
                        np.broadcast_to(value, broadcast_shape)
                    )

            return results
        if func is np.atleast_1d:
            values = tuple(
                value if isinstance(value, SymbolicProxy)
                else np.atleast_1d(value)
                for value in args
            )
            return values[0] if len(values) == 1 else list(values)
        if func is np.sum:
            values = args[0]
            if isinstance(values, SymbolicProxy):
                return values
            return SymbolicProxy(sp.Add(*(sp.sympify(v) for v in values)))
        return NotImplemented

    def ravel(self, order="C"):
        """Return a flattened symbolic proxy."""
        if isinstance(self.expression, sp.NDimArray):
            expression = sp.ImmutableDenseNDimArray(
                list(self.expression), (self.size,)
            )
        else:
            expression = self.expression

        return SymbolicProxy(expression, shape=(self.size,))

    @staticmethod
    def _array_apply(array, function):
        """Apply a scalar operation to every entry of a SymPy array.

        Build the dense array from a concrete list rather than NDimArray.applyfunc:
        some SymPy versions pass a one-shot ``map`` to the array constructor,
        which is incompatible with Python versions whose constructor requires
        a sized sequence.
        """
        indices = itertools.product(
            *(range(length) for length in array.shape)
        )
        values = [function(array[index]) for index in indices]
        return sp.ImmutableDenseNDimArray(values, array.shape)

    def _binary(self, other, op):
        other_expression = (
            other.expression if isinstance(other, SymbolicProxy)
            else sp.sympify(other)
        )
        left_is_array = isinstance(self.expression, sp.NDimArray)
        right_is_array = isinstance(other_expression, sp.NDimArray)

        if left_is_array and right_is_array:
            if self.expression.shape != other_expression.shape:
                raise ValueError(
                    "Symbolic array operands must have matching shapes; "
                    f"got {self.expression.shape} and {other_expression.shape}."
                )
            result = sp.ImmutableDenseNDimArray(
                [
                    op(self.expression[index], other_expression[index])
                    for index in itertools.product(
                        *(range(length) for length in self.expression.shape)
                    )
                ],
                self.expression.shape,
            )
            shape = self.expression.shape
        elif left_is_array:
            # Broadcast a scalar right operand across every array element.
            result = self._array_apply(
                self.expression, lambda value: op(value, other_expression)
            )
            shape = self.expression.shape
        elif right_is_array:
            # Reverse operations (notably scalar / array) must broadcast the
            # scalar over each element while preserving operand order.
            result = self._array_apply(
                other_expression, lambda value: op(self.expression, value)
            )
            shape = other_expression.shape
        else:
            result = op(self.expression, other_expression)
            shape = self.shape
        return SymbolicProxy(result, shape=shape)

    def _condition(self, other, cond):
        other_expression = (
            other.expression if isinstance(other, SymbolicProxy)
            else sp.sympify(other)
        )
        left_is_array = isinstance(self.expression, sp.NDimArray)
        right_is_array = isinstance(other_expression, sp.NDimArray)

        def indicator(left, right):
            return sp.Piecewise((1, cond(left, right)), (0, True))

        if left_is_array and right_is_array:
            if self.expression.shape != other_expression.shape:
                raise ValueError(
                    "Symbolic array operands must have matching shapes; "
                    f"got {self.expression.shape} and {other_expression.shape}."
                )
            result = sp.ImmutableDenseNDimArray(
                [
                    indicator(self.expression[index], other_expression[index])
                    for index in itertools.product(
                        *(range(length) for length in self.expression.shape)
                    )
                ],
                self.expression.shape,
            )
            shape = self.expression.shape
        elif left_is_array:
            result = self._array_apply(
                self.expression, lambda value: indicator(value, other_expression)
            )
            shape = self.expression.shape
        elif right_is_array:
            result = self._array_apply(
                other_expression, lambda value: indicator(self.expression, value)
            )
            shape = other_expression.shape
        else:
            result = indicator(self.expression, other_expression)
            shape = self.shape
        return SymbolicProxy(result, shape=shape)

    def _logical(self, other, operator):
        """Combine symbolic Boolean/indicator expressions elementwise."""
        def as_boolean(value):
            # Comparisons are represented as 0/1 Piecewise expressions.
            # Convert them back to SymPy Boolean expressions for And/Or.
            if value in (sp.true, sp.false):
                return value
            return sp.Ne(value, 0)

        return self._condition(
            other,
            lambda left, right: operator(as_boolean(left), as_boolean(right)),
        )

    def __and__(self, other):
        return self._logical(other, sp.And)

    def __rand__(self, other):
        return self._logical(other, lambda left, right: sp.And(right, left))

    def __or__(self, other):
        return self._logical(other, sp.Or)

    def __ror__(self, other):
        return self._logical(other, lambda left, right: sp.Or(right, left))

    def __invert__(self):
        def negate(value):
            return sp.Piecewise(
                (1, sp.Not(sp.Ne(value, 0))),
                (0, True),
            )

        if isinstance(self.expression, sp.NDimArray):
            result = self._array_apply(self.expression, negate)
        else:
            result = negate(self.expression)
        return SymbolicProxy(result, shape=self.shape)

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
        return SymbolicProxy(-self.expression, shape=self.shape)

    def __pos__(self):
        return SymbolicProxy(self.expression, shape=self.shape)

    def __abs__(self):
        return SymbolicProxy(sp.Abs(self.expression), shape=self.shape)

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
                return SymbolicProxy(self.symbols[col], shape=(1,))
        elif isinstance(key, int):
            # TODO: Do I need this mode, or is that invalid for the
            # rest of qmcpy?
            return SymbolicProxy(self.symbols[key], shape=(1,))
        raise NotImplementedError(f"Unsupported indexing: {key}")


@contextmanager
def monkeypatch_for_symbolic():
    """Monkeypatch out functions that don't work well with the symbolic conversion."""

    original_asarray = np.asarray
    def asarray_to_symbolic(value, dtype=None, copy=None):
        # scipy.stats calls np.asarray on its input before evaluating the
        # distribution. Preserve symbolic proxies instead of coercing them
        # into object arrays; retain NumPy's normal behavior for other inputs.
        if isinstance(value, SymbolicProxy):
            return value
        if copy is None:
            return original_asarray(value, dtype=dtype)
        return original_asarray(value, dtype=dtype, copy=copy)

    mapping = {
        np.asarray: (
            np,
            asarray_to_symbolic,
        ),
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
    with monkeypatch_for_symbolic():
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
