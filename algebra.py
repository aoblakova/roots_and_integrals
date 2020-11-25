import numpy.polynomial.polynomial as poly
import numpy.linalg as linalg
import numpy as np
import integrate


def find_sigmas_with_signs_from_roots(roots):
    """Find (-1)^k * sigma_k, k = 0, ..., n,
    where  sigma_k = \\sum_{1 <= i_1 < ... < i_k <= n} z_{i_1} * ... * z_{i_k}
    for  k = 0, ..., n  using  'roots' = [z_1,..., z_n].
    It is equivalent to returning coefficients of polynomial
    \\prod_i (z - z_i)  in reversed order."""
    return poly.polyfromroots([root for root in roots])[::-1]


def find_zetas_from_roots(roots, m):
    """Find first 'm' complete homogeneous polynomials of 'roots', i.e.,
    find  zeta_k = \\sum_{1 <= i_1 <= ... <= i_k <= n} z_{i_1} * ... * z_{i_k}
    for  k = 0, ..., m-1  from  'roots' = [z_1,..., z_n].
    It is equivalent to returning first 'm' coefficients of polynomial
    \\prod_i \\sum_{k=0}^{m-1} z_i^k * z^k."""
    # array of polynomials  \\sum_{k=0}^{m-1} z_i^k * z^k  for each root  z_i:
    polynomials = [[1.] + [0] * (m - 1) for _ in roots]
    for i in range(1, m):
        for index, root in enumerate(roots):
            polynomials[index][i] = polynomials[index][i - 1] * root

    # polynomial  \\prod_i \\sum_{k=0}^{m-1} z_i^k * z^k
    one_polynomial = [1]
    for polynomial in polynomials:
        one_polynomial = poly.polymul(one_polynomial, polynomial)
    return one_polynomial[:m]


def find_sigmas_from_etas(etas):
    """Compute sigmas from etas using
    k * sigma_k = \\sum_{j=1}^k (-1)^{j+1} sigma_{k-j} etas[j],  k = 1, ..., n,  where  len(etas) = n + 1."""
    n = len(etas) - 1
    sigmas = [0.] * (n + 1)
    sigmas[0] = 1.
    for k in range(1, n + 1):
        sigmas[k] = sum([(-1) ** (j + 1) * sigmas[k - j] * etas[j] for j in range(1, k + 1)]) / float(k)
    return sigmas


def find_zetas_from_sigmas(sigmas, m):
    """Find first m complete symmetrical polynomials  zetas  using 'sigmas'.
    For this we use:
    zeta_k = \\sum_{i=1}^k (-1)^(i+1) sigmas[i] zetas_{k-i}."""
    zetas = [0.] * m
    zetas[0] = 1.
    for k in range(1, m):
        zetas[k] = sum([sigmas[i] * zetas[k - i] * (-1) ** (i + 1) for i in range(1, min(len(sigmas), k + 1))])
    return zetas


def add_trailing_zeros(function, m):
    """Add  m - len(function)  zeros at the end of 'function',
    which is an array of coefficients in its Taylor expansion.
    If 'm' is less than the length of function, reshape function to 'm' elements."""
    if len(function) < m:
        return np.concatenate((function, [0] * (m - len(function))))
    return function[:m]


def find_transformations(zetas, functions):
    """Find matrix, where (i, j)-element of the matrix is
    the (i, n)-transformation of polynomial functions[j] for  i = 0, ..., n-1,  i.e.,
    find matrix of  \\sum_{i=k}^{len(functions[j]) - 1} functions[j][i] * zetas[i - k]."""
    # number of the roots
    n = len(functions) - 1
    m = len(zetas)

    # add trailing zeros if required
    functions = [add_trailing_zeros(function, m) for function in functions]

    # make the matrix of 'zetas' such that zetas_matrix[i, j] = z_{i - j} with z_{i} = 0 if i < 0
    zetas_matrix = np.zeros(shape=(m, n), dtype=complex)
    for i in range(min(n, m)):
        zetas_matrix[i:, i] = zetas[:m - i]

    # the result can be represented as matrix  zetas_matrix^T * functions^T   or   (functions * zetas_matrix)^T
    return np.transpose(np.matmul(functions, zetas_matrix))


def find_m_bar(roots, functions, first_row):
    """Find matrix  \\bar M  of 'functions', which are polynomials.
    \\bar M[0, :] = first_row,  \\bar M [i, j]  is (i - 1, n)-transformation of functions[j] for  i = 1, ..., n,
    where n is the number of roots and equals the number of functions minus 1."""
    n = len(roots)  # number of roots   (\\hat z_1, ..., \\hat z_n)
    if len(functions) != n + 1:
        raise Exception("The number of functions, %d, is not equal to the number of roots, %d, plus 1."
                        % (len(functions), n))

    m = max([len(function) for function in functions])  # the number of required zetas
    zetas = find_zetas_from_roots(roots, m)  # find required zetas

    return np.concatenate(([first_row], find_transformations(zetas, functions)))


def find_m_bar_integral(denominator, functions, first_row, epsilon):
    """Using contour integrals, find matrix  \\bar M  of 'functions', which are polynomials.
    \\bar M[0, :] = first_row,  \\bar M [i, j]  is (i - 1, n)-transformation of functions[j] for  i = 1, ..., n,
    where n is the number of roots (\\hat z_1, ..., \\hat z_n) and equals the number of functions minus 1."""
    n = len(functions) - 1
    etas = integrate.find_etas(denominator, n, epsilon)

    m = max([len(function) for function in functions])  # the number of required zetas
    zetas = find_zetas_from_sigmas(find_sigmas_from_etas(etas), m)
    return np.concatenate(([first_row], find_transformations(zetas, functions)))


def find_coefficients_bar(roots, functions, first_row, right_coefficient):
    """Find solution to \\bar M x = (right_coefficient, 0, ..., 0)^T, where
    \\bar M[0, :] = first_row,  \\bar M [i, j]  is (i - 1, n)-transformation of functions[j] for  i = 1, ..., n,
    and n is the number of roots."""
    m_bar = find_m_bar(roots, functions, first_row)
    return linalg.solve(m_bar, [right_coefficient] + [0] * len(roots))


def find_coefficients_bar_integral(denominator, functions, first_row, right_coefficient, epsilon):
    """Find solution to \\bar M x = (right_coefficient, 0, ..., 0)^T, where
    \\bar M[0, :] = first_row,  \\bar M [i, j]  is (i - 1, n)-transformation of functions[j] for  i = 1, ..., n,
    and  n  is the number of roots."""
    m_bar = find_m_bar_integral(denominator, functions, first_row, epsilon)
    return linalg.solve(m_bar, [right_coefficient] + [0] * (len(functions) - 1))


def find_coefficients(roots, functions, first_row, right_coefficient):
    """Find solution to M x = (right_coefficient, 0, ..., 0)^T, where
    M[0, :] = first_row,  M [i, j]  is the value of functions[j] at roots[i - 1] = \\hat z_i  for  i = 1, ..., n,
    and  n  is the number of the roots."""
    matrix = np.concatenate(([first_row],
                             [[poly.polyval(root, function) for function in functions] for root in roots]))
    return linalg.solve(matrix, [right_coefficient] + [0] * len(roots))
