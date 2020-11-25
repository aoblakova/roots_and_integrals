from scipy.integrate import quad
import numpy.polynomial.polynomial as poly
import numpy as np
import math


SMALL_EPS = 10 ** (-12)


def complex_from_angle(phi, radius):
    """Return the complex number with argument 'phi' and absolute value 'radius'."""
    return complex(math.cos(phi) * radius, math.sin(phi) * radius)


def integrand(phi, num_func, den_func, radius):
    """Compute the value of the integrand  z * num_func(z) / den_func(z),  where  z = z(phi) = radius * e^{i phi}."""
    x = complex_from_angle(phi, radius)
    return (x * poly.polyval(x, num_func) / poly.polyval(x, den_func)).real


def integrate(num_func, den_func, radius, epsilon):
    """Integrate the function   num_func(z) / ( 2 * pi * i * den_func(z)) dz =
    z num_func(z) / ( 2 * pi * den_func(z)) d phi  along the circle  S_radius
    with an absolute error tolerance of  'epsilon'. Return the real part of the integral."""
    integral_real, _ = quad(integrand, -math.pi, math.pi, args=(num_func, den_func, radius), epsabs=epsilon)
    return integral_real / (2 * math.pi)


def find_good_radius(polynom):
    """Find a good radius, i.e., such 'radius' that there are no roots in  R_radius = {z: 1 < |z| < radius}.
    'polynom' is such a polynomial that it has only one zero  z_{-1}  in  (1, \\infty),  polynom(1) = 0,
    polynom'(1) > 0, and there are no zeroes in  R_{z_{-1}}.
    Return 1 + 2^(-k) such that  'k'  is he first integer for which  polynom(1 + 2^(-k)) > 0.  """
    radius = 1.5
    while poly.polyval(radius, polynom) <= 0:
        radius = (radius + 1) / 2.
        if radius < 1. + SMALL_EPS:
            print("Warning: haven't found good radius by dividing!")
    return radius


def find_etas(polynom, n, epsilon):
    """Find  eta_k = \\sum_(j=1)^n \\hat z_j^k  for  k = 0, ..., n - 1.
    eta_k = -1 + 1 / (2 * pi * i) * \\int_{S_{1 + epsilon}} polynom'(z) z^k / polynom(z) dz."""
    etas = [float(n)]  # eta_0 = n
    radius = find_good_radius(polynom)
    der_zk = poly.polyder(polynom)  # polynom'(z) z^k,  starting with  k = 0
    for k in range(1, n + 1):
        der_zk = np.concatenate(([0], der_zk))
        etas.append(-1 + integrate(der_zk, polynom, radius, epsilon))
    return etas
