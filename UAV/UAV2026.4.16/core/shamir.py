# core/shamir.py
from __future__ import annotations

import random
from typing import List, Tuple


def mod_inverse(a: int, p: int) -> int:
    """Return modular inverse of a mod p, assuming p is prime."""
    return pow(a, -1, p)


def make_polynomial(secret: int, t: int, prime: int) -> List[int]:
    """
    Build a random polynomial of degree t-1:
        f(x) = secret + a1*x + a2*x^2 + ...
    """
    if t < 2:
        raise ValueError("Threshold t must be at least 2.")
    coeffs = [secret] + [random.randrange(1, prime) for _ in range(t - 1)]
    return coeffs


def evaluate_polynomial(coeffs: List[int], x: int, prime: int) -> int:
    """Evaluate polynomial at x under mod prime."""
    result = 0
    power = 1
    for coeff in coeffs:
        result = (result + coeff * power) % prime
        power = (power * x) % prime
    return result


def split_secret(secret: int, n: int, t: int, prime: int) -> List[Tuple[int, int]]:
    """
    Split secret into n shares with threshold t.
    Return shares as list of (x, y).
    """
    if not (1 < t <= n):
        raise ValueError("Require 1 < t <= n.")
    coeffs = make_polynomial(secret, t, prime)
    shares = []
    for x in range(1, n + 1):
        y = evaluate_polynomial(coeffs, x, prime)
        shares.append((x, y))
    return shares


def lagrange_interpolate_at_zero(points: List[Tuple[int, int]], prime: int) -> int:
    """
    Recover f(0) from shares using Lagrange interpolation.
    """
    if len(points) == 0:
        raise ValueError("At least one point is required.")

    secret = 0
    for i, (x_i, y_i) in enumerate(points):
        numerator = 1
        denominator = 1
        for j, (x_j, _) in enumerate(points):
            if i == j:
                continue
            numerator = (numerator * (-x_j)) % prime
            denominator = (denominator * (x_i - x_j)) % prime
        lagrange_coeff = numerator * mod_inverse(denominator, prime)
        secret = (secret + y_i * lagrange_coeff) % prime

    return secret


def recover_secret(points: List[Tuple[int, int]], prime: int) -> int:
    """Recover secret from shares."""
    return lagrange_interpolate_at_zero(points, prime)


def lagrange_coefficients_at_zero(xs: list[int], prime: int) -> list[int]:
    coeffs = []
    for i, x_i in enumerate(xs):
        numerator = 1
        denominator = 1
        for j, x_j in enumerate(xs):
            if i == j:
                continue
            numerator = (numerator * (-x_j)) % prime
            denominator = (denominator * (x_i - x_j)) % prime
        coeff = (numerator * mod_inverse(denominator, prime)) % prime
        coeffs.append(coeff)
    return coeffs


def recover_group_element(
    points: list[tuple[int, object]],
    prime: int,
    scalar_mul_fn,
    point_add_fn,
):
    """
    恢复群元素形式的 Σ λ_i * P_i
    points: [(x_i, P_i)]
    """
    if len(points) == 0:
        raise ValueError("At least one point is required.")

    xs = [x for x, _ in points]
    lambdas = lagrange_coefficients_at_zero(xs, prime)

    acc = None
    for lam, (_, point_i) in zip(lambdas, points):
        term = scalar_mul_fn(point_i, lam)
        acc = term if acc is None else point_add_fn(acc, term)

    return acc