from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import sympy as sp
from sympy import Rational
from scripts.remove_trailing_whitespace import remove_trailing_whitespace

from qmcpy.util import (
    _univ_repr,
    NotYetImplemented,
    ParameterError,
    MethodImplementationError,
)
from qmcpy.util.data import Data
from qmcpy.util.dig_shift_invar_ops import to_bin, to_float, weighted_walsh_funcs
from qmcpy.util.latnetbuilder_linker import latnetbuilder_linker
from qmcpy.util.shift_invar_ops import bernoulli_poly
from qmcpy.util.torch_numpy_ops import get_npt
from qmcpy.util.transforms import (
    insert_batch_dims,
    parse_assign_param,
    tf_exp,
    tf_exp_inv,
    tf_square,
    tf_square_inv,
    tf_identity,
    tf_explinear,
    tf_explinear_inv,
)
from qmcpy.util.symbolic import max_norm


class TestTorchNumpyOps(unittest.TestCase):
    """Tests for torch_numpy_ops.get_npt()."""

    def test_get_npt_numpy_array(self):
        """get_npt should return np for numpy arrays."""
        x = np.array([1.0, 2.0, 3.0])
        npt = get_npt(x)
        self.assertEqual(npt, np)

    def test_get_npt_torch_tensor(self):
        """get_npt should return torch for torch tensors."""
        try:
            import torch
            x = torch.tensor([1.0, 2.0, 3.0])
            npt = get_npt(x)
            self.assertEqual(npt, torch)
        except ImportError:
            self.skipTest("torch not installed")


class TestTransforms(unittest.TestCase):
    """Tests for transform functions."""

    def test_insert_batch_dims(self):
        """insert_batch_dims should add batch dimensions."""
        param = np.array([1.0, 2.0, 3.0])
        result = insert_batch_dims(param, ndims=2, k=0)
        self.assertEqual(result.shape, (1, 1, 3))

    def test_insert_batch_dims_middle(self):
        """insert_batch_dims should work with k > 0."""
        param = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = insert_batch_dims(param, ndims=1, k=1)
        self.assertEqual(result.shape, (2, 1, 2))

    def test_tf_exp(self):
        """tf_exp should compute element-wise exponential."""
        x = np.array([0.0, 1.0, 2.0])
        result = tf_exp(x)
        expected = np.exp(x)
        np.testing.assert_allclose(result, expected)

    def test_tf_exp_inv(self):
        """tf_exp_inv should compute element-wise logarithm."""
        x = np.array([1.0, np.e, np.e**2])
        result = tf_exp_inv(x)
        expected = np.log(x)
        np.testing.assert_allclose(result, expected)

    def test_tf_square(self):
        """tf_square should square elements."""
        x = np.array([1.0, 2.0, 3.0])
        result = tf_square(x)
        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_array_equal(result, expected)

    def test_tf_square_inv(self):
        """tf_square_inv should take square root."""
        x = np.array([1.0, 4.0, 9.0])
        result = tf_square_inv(x)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_tf_identity(self):
        """tf_identity should return input unchanged."""
        x = np.array([1.0, 2.0, 3.0])
        result = tf_identity(x)
        np.testing.assert_array_equal(result, x)

    def test_tf_explinear(self):
        """tf_explinear should apply explinear transform."""
        x = np.array([0.0, 1.0, 2.0])
        result = tf_explinear(x)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_tf_explinear_inv(self):
        """tf_explinear_inv should apply inverse explinear transform."""
        x = np.array([0.5, 1.0, 2.0])
        result = tf_explinear_inv(x)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_tf_explinear_inv_large(self):
        """tf_explinear_inv should handle large values correctly."""
        x = np.array([35.0, 100.0])
        result = tf_explinear_inv(x)
        # For large x, should return approximately x
        np.testing.assert_allclose(result[1], x[1], rtol=1e-10)

    def test_parse_assign_param_numpy_constraints(self):
        """parse_assign_param should respect numpy constraints."""
        raw = parse_assign_param(
            pname="sigma",
            param=2.0,
            shape_param=[1],
            requires_grad_param=False,
            tfs_param=(tf_identity, tf_identity),
            endsize_ops=[1, 2],
            constraints=["POSITIVE"],
            torchify=False,
            npt=np,
            nptkwargs={},
        )
        np.testing.assert_allclose(raw, np.array([2.0]))

        with self.assertRaises(AssertionError):
            parse_assign_param(
                pname="k",
                param=1.5,
                shape_param=[1],
                requires_grad_param=False,
                tfs_param=(tf_identity, tf_identity),
                endsize_ops=[1],
                constraints=["INTEGER"],
                torchify=False,
                npt=np,
                nptkwargs={},
            )


class TestDigShiftInvarOps(unittest.TestCase):
    """Tests for digital shift invariant utility helpers."""

    def test_digital_shift_roundtrip_and_dtype_errors(self):
        """to_bin and to_float should roundtrip float inputs."""
        xf = np.array([0.0, 0.125, 0.25, 0.875], dtype=float)
        xb = to_bin(xf, t=3)
        self.assertEqual(xb.dtype, np.uint64)
        x_back = to_float(xb, t=3)
        np.testing.assert_allclose(x_back, xf)

        with self.assertRaises(ParameterError):
            to_bin(np.array([1 + 2j], dtype=np.complex128), t=3)

    def test_weighted_walsh_funcs_and_bernoulli_poly_input_checks(self):
        """weighted_walsh_funcs and bernoulli_poly should validate inputs."""
        xb = np.array([[0, 1], [2, 3]], dtype=np.uint64)
        y = weighted_walsh_funcs(alpha=2, xb=xb, t=2)
        self.assertEqual(y.shape, xb.shape)
        self.assertTrue(np.isfinite(y).all())

        with self.assertRaises(AssertionError):
            weighted_walsh_funcs(alpha=5, xb=xb, t=2)

        yb = bernoulli_poly(2, np.array([0.0, 0.5, 1.0]))
        self.assertEqual(yb.shape, (3,))
        with self.assertRaises(AssertionError):
            bernoulli_poly(11, np.array([0.1]))


class TestLatnetbuilderLinker(unittest.TestCase):
    """Tests for latnetbuilder linker utility."""

    def test_latnetbuilder_linker_not_implemented(self):
        """latnetbuilder_linker should currently raise NotYetImplemented."""
        with self.assertRaises(NotYetImplemented):
            latnetbuilder_linker()


class TestUnivRepr(unittest.TestCase):
    """Tests for _univ_repr utility function."""

    @staticmethod
    def _make_mock(**attrs):
        """helper function to create a mock object with specified attributes."""
        obj = type("MockObject", (), {})()
        for key, value in attrs.items():
            setattr(obj, key, value)
        return obj

    def test_univ_repr_scalar_values(self):
        """_univ_repr should format scalar values nicely."""
        obj = self._make_mock(n_samples=128, dimension=5)
        result = _univ_repr(obj, "MockClass", ["n_samples", "dimension"])
        self.assertIn("MockClass", result)
        self.assertIn("n_samples", result)
        self.assertIn("dimension", result)

    def test_univ_repr_power_of_two(self):
        """_univ_repr should recognize powers of 2."""
        obj = self._make_mock(n=512)  # 2^9
        result = _univ_repr(obj, "Test", ["n"])
        self.assertIn("2^(9)", result)

    def test_univ_repr_array_values(self):
        """_univ_repr should handle numpy arrays."""
        obj = self._make_mock(data=np.array([1.0, 2.0, 3.0]))
        result = _univ_repr(obj, "Test", ["data"])
        self.assertIn("Test", result)

    def test_univ_repr_list_values(self):
        """_univ_repr should handle list values."""
        obj = self._make_mock(seeds=[1, 2, 3])
        result = _univ_repr(obj, "Test", ["seeds"])
        self.assertIn("Test", result)


class TestData(unittest.TestCase):
    """Tests for Data class."""

    @staticmethod
    def _make_data(parameters=None, **attrs):
        data = Data(parameters or [])
        data.time_integrate = 0.0
        for key, value in attrs.items():
            setattr(data, key, value)
        return data

    def test_data_init(self):
        """Data should initialize with parameters list."""
        d = Data(["param1", "param2"])
        self.assertEqual(d.parameters, ["param1", "param2"])

    def test_data_repr_empty(self):
        """Data.__repr__ should generate proper string with empty params."""
        d = self._make_data()
        result = str(d)
        self.assertIn("Data", result)

    def test_data_repr_with_values(self):
        """Data.__repr__ should show attributes when parameters are set."""
        d = self._make_data(parameters=["test_param"], test_param=42)
        d.time_integrate = 0.5
        result = str(d)
        self.assertIn("Data", result)
        self.assertIn("test_param", result)

    def test_data_repr_with_optional_attributes(self):
        """Data.__repr__ should show optional related objects if set."""
        for attr, value in [
            ("stopping_crit", "test_crit"),
            ("integrand", "test_integrand"),
            ("true_measure", "test_measure"),
            ("discrete_distrib", "test_distrib"),
        ]:
            with self.subTest(attr=attr):
                d = self._make_data(**{attr: value})
                result = str(d)
                self.assertIn(value, result)


class TestExceptionsWarnings(unittest.TestCase):
    """Tests for exception and warning classes."""

    class _TestClass:
        pass

    def test_parameter_error_instantiation(self):
        """ParameterError should be instantiable with message."""
        err = ParameterError("Test error message")
        self.assertIn("Test error message", str(err))

    def test_parameter_error_inheritance(self):
        """ParameterError should inherit from Exception."""
        err = ParameterError("Test")
        self.assertIsInstance(err, Exception)

    def test_parameter_error_raised(self):
        """ParameterError should be raised and caught properly."""
        with self.assertRaises(ParameterError):
            raise ParameterError("test")

    def test_method_implementation_error_instantiation(self):
        """MethodImplementationError needs subclass and method_name."""
        err = MethodImplementationError(self._TestClass(), "test_method")
        self.assertIn("_TestClass", str(err))
        self.assertIn("test_method", str(err))

    def test_method_implementation_error_inheritance(self):
        """MethodImplementationError should inherit from Exception."""
        err = MethodImplementationError(self._TestClass(), "test")
        self.assertIsInstance(err, Exception)

    def test_method_implementation_error_raised(self):
        """MethodImplementationError should be raised and caught."""
        with self.assertRaises(MethodImplementationError):
            raise MethodImplementationError(self._TestClass(), "test")


class TestRemoveTrailingWhitespace(unittest.TestCase):
    def test_preserves_python_string_contents_and_line_endings(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "example.py"
            original = b'"""expected output  \r\n"""\r\nx = "value"  \r\ny = 2\t\n'
            expected = b'"""expected output  \r\n"""\r\nx = "value"\r\ny = 2\n'
            path.write_bytes(original)

            self.assertTrue(remove_trailing_whitespace(path, check=True))
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(remove_trailing_whitespace(path, check=False))
            self.assertEqual(path.read_bytes(), expected)
            self.assertFalse(remove_trailing_whitespace(path, check=True))

    def test_preserves_multiline_f_string_trailing_spaces(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "example.py"
            original = b'x = f"""hello  \nworld  \n"""\nvalue = 1  \n'
            expected = b'x = f"""hello  \nworld  \n"""\nvalue = 1\n'
            path.write_bytes(original)

            self.assertTrue(remove_trailing_whitespace(path, check=False))
            self.assertEqual(path.read_bytes(), expected)



class TestSymbolicSolver(unittest.TestCase):
    """Test symbolic solver utility helpers for finding unnormalized function maximum and area."""

    def test_basic_1d_beta(self):
        """Check that the symbolic solver is working on a known simple
        example, Beta(5,7).

        """

        def psi(x):
            return x[:, 0] ** 4 * (1 - x[:, 0]) ** 6
        dimension = 1
        max_at = (Rational(4,10),)
        upper_bound = psi(np.array([max_at]))
        density_integral = Rational(1,2310)

        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        self.assertEqual(density_integral, calc_density)

        
    def test_basic_2d_beta(self):
        """Check that the symbolic solver is working on a known simple
        2D example.

        """
        
        def psi(x):
            return (
                x[:, 0] ** 4
                * (1 - x[:, 0]) ** 6
                * x[:, 1] ** 2
                * (1 - x[:, 1]) ** 3
            )
        dimension = 2
        max_at = (Rational(4,4+6),Rational(2,2+3))
        upper_bound = psi(np.array([max_at]))
        density_integral = Rational(1,2310 * 60)
        
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)
        
        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        self.assertEqual(density_integral, calc_density)

        
    def test_maximum_on_1d_boundary(self):
        """The maximum occurs at x=1, which is not at an interior
        critical point.

        """
        
        def psi(x):
            return x[:, 0] ** 2
        dimension = 1
        max_at = (1,)
        upper_bound = 1
        density_integral = Rational(1,3)
        
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        self.assertEqual(density_integral, calc_density)

        
    def test_maximum_on_2d_corner(self):
        """The maximum occurs at (1, 0), on a corner of the square."""
        
        def psi(x):
            return x[:, 0] ** 2 * (1 - x[:, 1]) ** 3
        dimension = 2
        max_at = (1,0)
        upper_bound = 1
        density_integral = Rational(1,12)
        
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        self.assertEqual(density_integral, calc_density)


    def test_maximum_on_2d_face(self):
        """The maximum occurs at (0.5, 0), on a face of the square."""
        def psi(x):
            return 4 - x[:,0]**2 - (.5 - x[:,1])**2
        dimension = 2
        max_at = (0, 0.5)
        upper_bound = 4
        density_integral = 3+Rational(1,4)+Rational(1,3)
        
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        for m,l in zip(max_at, upper_location):
            self.assertEqual(m, l)
        self.assertEqual(upper_bound, calc_upper)
        # TODO: Why is this not exact?
        self.assertAlmostEqual(density_integral, calc_density, delta=1e-10)


    def test_maximum_outside_of_domain(self):
        """The global maximum critical point is outside of the domain,
        at x=-1.
        
        """
        
        def psi(x):
            return (x[:, 0]+3) ** 4 * (2 - x[:, 0]) ** 6
        dimension = 1
        max_at = (0,)
        upper_bound = 5184
        density_integral = Rational(951047,462)
        
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        self.assertEqual(density_integral, calc_density)


    def test_constant_density(self):
        """A constant density has no critical points.

        """
        
        def psi(x):
            return np.full(x.shape[0], 3.0)
        dimension=1
        upper_bound=3.0
        density_integral=3.0
        
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)
        
        self.assertEqual(upper_bound, calc_upper)
        self.assertEqual(density_integral, calc_density)

        
    def test_multiple_boundary_candidates(self):
        """There are candidate values for the maximum in both the
        interior and on the boundary.

        """
        
        def psi(x):
            return 1 + x[:, 0] * (1 - x[:, 0])
        dimension=1
        max_at = (Rational(1,2),)
        upper_bound=Rational(5,4)
        density_integral = 1+Rational(1,6)
        
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        self.assertEqual(density_integral, calc_density)


    def test_density_with_no_finite_integral(self):
        """The integral diverges at x=0."""
        
        def psi(x):
            return 1 / x[:, 0]
        dimension = 1
        with self.assertRaises(ParameterError):
            max_norm(psi, dimension)


    def test_density_with_no_finite_maximum(self):
        """The density is integrable but unbounded at x=0."""
        
        def psi(x):
            return (x[:, 0]) ** (-1/2)
        dimension = 1
        with self.assertRaises(ParameterError):
            max_norm(psi, dimension)


    def test_negative_density(self):
        """A negative density must be rejected.

        Note: This test cannot be exhaustive, if the density is not
        negative at one of the projected critical points, it will not
        be detected.

        """
        
        def psi(x):
            return x[:, 0] - 1/2
        dimension = 1
        with self.assertRaises(ParameterError):
            max_norm(psi, dimension)

        
    def test_zero_integral(self):
        """A density that is identically zero is invalid."""
        
        def psi(x):
            return 0
        dimension = 1
        with self.assertRaises(ParameterError):
            max_norm(psi, dimension)


    def test_zero_maximum(self):
        """A non-positive density cannot define a valid rejection sampler."""

        def psi(x):
            return -(x[:, 0] - 1/2) ** 2
        dimension = 1
        with self.assertRaises(ParameterError):
            max_norm(psi, dimension)


    def test_multiple_equal_global_maxima(self):
        """All global maxima have the same value, find that value."""
        def psi(x):
            return 1 - x[:, 0] ** 2 * (1 - x[:, 0]) ** 2
        dimension = 1

        # TODO: This test actually has a greater global critical point
        # maxima in the complex domain?
        
        # Local extrema occur at x=0 and x=1, with value 1, and at
        # x=1/2, with value 15/16. The global maximum is therefore 1.
        upper_bound=1
        density_integral=1-Rational(1, 30)

        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        # assert In upper_location [0,1]
        self.assertEqual(upper_bound, calc_upper)
        self.assertEqual(density_integral, calc_density)
        

    def test_multiple_different_local_maxima_strictly_within_domain(self):
        """Several local maxima have different values, find the correct one."""
        
        def psi(x):
            return (3-(x[:, 0]-0.1) ** 2) * (2-(0.9 - x[:, 0]) ** 2)
        dimension = 1
        
        # Local extrema occur at x=0.1 and x=0.9, with values _ and _ respectively.
        upper_bound=max(
            psi(np.array([[0.1]])),
            psi(np.array([[0.9]])),
        )
        #TODO: density_integral=Rational(151, 15)

        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        # TODO: self.assertEqual(max_at, upper_location)
        # TODO: self.assertEqual(upper_bound, calc_upper)
        # TODO: self.assertEqual(density_integral, calc_density)
        

    def test_scipy_stats_support(self):
        """Use a function from scipy stats."""
        from scipy.integrate import quad
        from scipy.stats import norm
        def psi(x):
            return norm.pdf(x[:, 0], loc=0, scale=1)
        dimension=1
        max_at = (0,)
        upper_bound=sp.sqrt(1/(2*sp.pi))
        density_integral=sp.erf(sp.sqrt(2)/2)/2
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)
        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        self.assertEqual(density_integral, calc_density)
        
    def test_numpy_function_support(self):
        """Verify that the numpy functions are handled correctly."""
        from scipy.integrate import quad
        functions = {
            'sqrt': np.sqrt,
            'sin': np.sin,
            'cos': np.cos,
        }
        for func_name in functions:
            with self.subTest(f"Verify numpy's {func_name=} is supported"):
                func=functions[func_name]
                def psi(x):
                    return func(x[:, 0])
                dimension=1
                max_at = 1 if func(1)>func(0) else 0
                max_at = (max_at,)
                upper_bound=func(max_at[0])
                density_integral=quad(lambda t: func(t), 0, 1)[0]
                [upper_location, calc_upper], calc_density = max_norm(psi, dimension)
                self.assertEqual(max_at, upper_location)
                self.assertAlmostEqual(upper_bound, calc_upper, delta=1e-10)
                self.assertAlmostEqual(density_integral, calc_density, delta=1e-10)

        
    def test_numpy_abs_function_support(self):
        """Verify that the numpy abs function is handled correctly."""
        
        def psi(x):
            return 1-np.abs(.5-x[:, 0])
        dimension=1
        max_at = (.5,)
        upper_bound=1
        density_integral=Rational(3,4)
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)
        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        # TODO: Why is this not exact?
        self.assertAlmostEqual(density_integral, calc_density, delta=1e-10)

        
    def test_insolvable_outside_region_1d(self):
        """The derivative involves cos(x^2), which is very hard to
        find the zeros of.  However, on [0,1], sin(x^2) is increasing,
        so the maximum is at x=1.

        """

        from scipy.integrate import quad
        def psi(x):
            return 2 + np.sin(x[:, 0] ** 2)
        dimension = 1
        max_at = (1,)
        upper_bound = sp.sin(1)+2
        # There is an exact answer; but, it's more complicated to
        # verify than reading this.
        density_integral = 2 + quad(lambda t: np.sin(t**2), 0, 1)[0]
        
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        self.assertAlmostEqual(density_integral, calc_density, delta=1e-10)


    def test_unsolvable_outside_region_2d(self):
        """The derivative involves cos(x^2), which is very hard to
        find the zeros of.  However, on [0,1], sin(x^2) is increasing,
        so the maximum is at x=1.

        """
        
        from scipy.integrate import dblquad
        def psi(x):
            return 1 + np.sin(x[:, 0] * x[:, 1])
        dimension = 2
        max_at = (1,1)
        upper_bound = sp.sin(1)+1
        density_integral = 1 + dblquad(
            lambda y, x: np.sin(x * y),
            0, 1,
            lambda x: 0,
            lambda x: 1,
        )[0]
        
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        self.assertAlmostEqual(density_integral, calc_density, delta=1e-10)


    def test_auto_piecewise_function(self):
        """Functions with piecewise branches are currently
        unavailable.

        """
        
        def psi(x):
            cond = ((3+x[:, 0]) < 3.5)
            return cond*x[:,0] + (1-cond)*(1-x[:,0])

        dimension = 1
        max_at = (0.5,)
        upper_bound = 0.5
        density_integral = Rational(1,4)
        
        [upper_location, calc_upper], calc_density = max_norm(psi, dimension)

        self.assertEqual(max_at, upper_location)
        self.assertEqual(upper_bound, calc_upper)
        # TODO: Why is this not exact?
        self.assertAlmostEqual(density_integral, calc_density, delta=1e-10)

        
    def test_auto_python_function(self):
        """Functions with calculations using code rather than logic
        are currently unavailable.

        """
        
        def psi(x):
            if x[0] < 0.5:
                return x[0]
            return 1-x[0]

        dimension = 1
        with self.assertRaises(NotYetImplemented):        
            max_norm(psi, dimension)


    def test_auto_hidden_python_function(self):
        """Functions with hidden branches (like catch)
        are forbidden, even if not used.

        """
        
        def psi(x):
            try:
                return np.sqrt(x+1)
            except:
                return 0

        dimension = 1
        
        with self.assertRaises(ParameterError):
            max_norm(psi, dimension)
if __name__ == "__main__":
    unittest.main()
