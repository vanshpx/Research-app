"""
tools/calculator_tool.py
-------------------------
Phase 6 — Safe mathematical expression evaluator using SymPy.

Use this tool whenever a numerical computation is required.
NEVER perform arithmetic mentally — delegate to this tool instead.

Supported operations:
  Arithmetic        2 * (5 + 7),  100 / 4,  2 ** 10
  Square roots      sqrt(25),  sqrt(2)
  Powers            2**8,  Rational(1,3)
  Trigonometry      sin(pi/2),  cos(0),  tan(pi/4)
  Logarithms        log(100, 10),  log(E)   (natural log)
  Constants         pi,  E

Security:
  eval() is NEVER used.  Expressions are parsed exclusively through
  SymPy's `sympify` with a restricted local namespace, so arbitrary
  Python cannot be executed even if the LLM produces a crafted string.

Future phases:
  Phase 8 — workspace tool can save computation history.
  Phase 9 — planner can chain calculator results into multi-step reasoning.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

# SymPy imports — all symbols that should be resolvable in expressions.
import sympy
from sympy import (
    E,
    Float,
    Integer,
    N,
    Rational,
    cos,
    log,
    pi,
    sin,
    sqrt,
    sympify,
    tan,
)
from sympy.core.sympify import SympifyError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe namespace
# ---------------------------------------------------------------------------

# Only these names are available when parsing an expression.
# This prevents __import__, open(), exec(), or any other Python builtin
# from being reachable through the expression string.
_SAFE_NAMESPACE: dict[str, Any] = {
    # Constants
    "pi": pi,
    "E": E,
    "e": E,          # common lowercase alias
    # Functions
    "sqrt": sqrt,
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "log": log,      # log(x) = natural log; log(x, base) = log base
    # Numeric helpers
    "Rational": Rational,
    "Integer": Integer,
    "Float": Float,
}

# ---------------------------------------------------------------------------
# Expression evaluator
# ---------------------------------------------------------------------------


def _evaluate(expression: str) -> str:
    """
    Parse and evaluate *expression* using SymPy.

    Steps:
      1. Sanitise the input string (strip whitespace, reject empty input).
      2. Parse via sympify() against the restricted namespace only.
      3. Numerically evaluate the result with N() for non-integer results.
      4. Return a clean string representation.

    Args:
        expression: Mathematical expression string.

    Returns:
        String representation of the result.

    Raises:
        ValueError: For empty, non-evaluatable, or forbidden expressions.
        SympifyError: For unparseable expressions (re-raised as ValueError).
    """
    expr_stripped = expression.strip()

    if not expr_stripped:
        raise ValueError("Expression is empty.")

    # Block any attempt to access Python builtins through the string.
    _reject_dangerous_tokens(expr_stripped)

    try:
        parsed = sympify(expr_stripped, locals=_SAFE_NAMESPACE, evaluate=True)
    except SympifyError as exc:
        raise ValueError(f"Could not parse expression: {exc}") from exc

    # If the result is still symbolic (e.g. sin(1) without numeric eval),
    # force a float approximation.
    if parsed.is_number:
        numeric = N(parsed, 15)  # 15 significant figures
        # Return as integer string if the result is a whole number.
        if numeric == int(numeric):
            return str(int(numeric))
        return str(numeric)

    # For symbolic results (e.g. sqrt(x)) return the simplified form.
    return str(parsed)


def _reject_dangerous_tokens(expr: str) -> None:
    """
    Raise ValueError if the expression contains obviously forbidden patterns.

    This is a defence-in-depth check on top of the restricted namespace.
    sympify with a safe namespace already blocks most attacks, but an
    explicit denylist makes the intent clear and provides a better error.

    Args:
        expr: Raw expression string from the LLM.

    Raises:
        ValueError: If a forbidden pattern is found.
    """
    forbidden = [
        "__",        # dunder access (__import__, __builtins__, …)
        "import",
        "exec",
        "eval",
        "open(",
        "os.",
        "sys.",
        "subprocess",
        "lambda",
    ]
    expr_lower = expr.lower()
    for token in forbidden:
        if token in expr_lower:
            raise ValueError(
                f"Expression contains a forbidden token: '{token}'. "
                "Only mathematical expressions are allowed."
            )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely using SymPy.

    Use this tool for ANY arithmetic or mathematical computation.
    Never compute numbers mentally — always delegate here.

    Supported:
      Arithmetic:    2*(5+7),  100/4,  2**10
      Square roots:  sqrt(25),  sqrt(2)
      Trigonometry:  sin(pi/2),  cos(0),  tan(pi/4)
      Logarithms:    log(100, 10),  log(E)
      Constants:     pi,  E

    Args:
        expression: A mathematical expression string.
                    Examples:
                      "2 * (5 + 7)"
                      "sqrt(25)"
                      "sin(pi / 2)"
                      "log(100, 10)"
                      "(0.8 * 0.9) / (0.8 + 0.9) * 2"   ← F1 score numerics

    Returns:
        The result as a string (integer or decimal as appropriate).
        On error, returns a descriptive error message string so the
        agent can report it cleanly rather than crashing.
    """
    logger.info("calculator — expression: %r", expression)

    try:
        result = _evaluate(expression)
        logger.info("calculator — result: %s", result)
        return result
    except ValueError as exc:
        logger.warning("calculator — invalid expression %r: %s", expression, exc)
        return f"Error: {exc}"
    except Exception as exc:
        logger.error("calculator — unexpected error for %r: %s", expression, exc)
        return f"Error: Could not evaluate expression. Details: {exc}"
