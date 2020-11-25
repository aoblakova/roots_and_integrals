import numpy.polynomial.polynomial as poly
import math
import cmath

EPS = 10 ** (-8)
LIMIT = 20000


def pick_right_roots(roots):
    """Return array of those elements of 'roots' that are inside unit disk with  1  at the first position."""
    roots_inside = [root for root in roots if abs(root) < 1 - EPS]
    near_1 = [root for root in roots if abs(root - 1) < EPS]
    if len(near_1) != 1:
        print(near_1, roots)
        raise ValueError("Root 1 has wrong multiplicity!")
    return near_1 + roots_inside


def find_roots_of_1(n):
    """Find complex n-roots of  1."""
    return [complex(math.cos(i * 2 * math.pi / n), math.sin(i * 2 * math.pi / n)) for i in range(n)]


def find_next_coefficient_c(k, previous_c, lamb, beta):
    """Return  c_k = lamb^(k-1) * (1 - lamb)^(k * beta - k + 1) * \\binom{k * beta}{k - 1} / k.
    To avoid large values of  \\binom  when 'k' is sufficiently big, and to decrease the runtime,
    the value is computed as:
    с_1 = (1-lamb)^beta,
    c_k = c_{k-1} * lamb * (1 - lamb)^(beta-1) * beta * \\prod_{j=1}^{beta-1} (beta * k - j) / (beta * k - k + 2 - j).
    """
    if k == 1:
        return (1. - lamb) ** beta
    result = previous_c * lamb * (1. - lamb) ** (beta - 1) * beta
    for j in range(1, beta):
        result *= (beta * k - j) / (beta * k - k + 2. - j)
    return result


def find_binomial_roots(lamb, n, beta, epsilon):
    """Find roots of equation  z^n = (lamb * z + 1 - lamb)^(n * beta)  using (16) from Janssen & van Leeuwaarden (2008):
    z_k ~ sum_{l=1}^{N} c_l w_k^l,
    where  c_l = lamb^(l-1) * (1-lamb)^(l * beta - l + 1) * binom{l * beta}{l - 1} / l.
    N = N(epsilon), which is the smallest value of  N  such that  1 - sum_{l=1}^{N} c_l < epsilon.
    """
    cs = [0]
    cs_sum = 0
    new_c = 1.
    iteration = 1
    while cs_sum + epsilon < 1.:
        new_c = find_next_coefficient_c(iteration, new_c, lamb, beta)
        cs.append(new_c)
        cs_sum += new_c
        if iteration >= LIMIT:
            # rate of convergence is slow
            print("Rate of convergence is too slow! After %d steps root z_0 is approximated by %f" % (LIMIT, cs_sum))
            raise Exception("Algorithm failed to find the roots in %d steps." % LIMIT)
        iteration += 1
    ws = find_roots_of_1(n)
    return poly.polyval(ws, cs)


def find_binomial_roots_beta_2(lamb, n):
    """Find roots of equation  z^n = (lamb * z + 1 - lamb)^(2 * n)  by separating it in quadratic equations
    z = w (lamb * z + 1 - lamb)^2,  where  w  is a n-root of  1."""
    roots = []
    ws = find_roots_of_1(n)
    for w in ws:
        roots.append((1 - 2 * lamb * (1 - lamb) * w + cmath.sqrt(1 - 4 * lamb * (1 - lamb) * w)) / (2 * w * lamb ** 2))
        roots.append((1 - 2 * lamb * (1 - lamb) * w - cmath.sqrt(1 - 4 * lamb * (1 - lamb) * w)) / (2 * w * lamb ** 2))
    return pick_right_roots(roots)
