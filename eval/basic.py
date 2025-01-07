import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ddx_ai import DDx

DDx("Nondimensionalize the polynomial\\[a_{1} x^{25} + a_{2} x + a_{3}\\]into one of the form $\\epsilon y^{25} + y^{1} + 1. $Express $\\epsilon$ as a function of $a_1$, $a_2$, and $a_3.$", verbose=True)