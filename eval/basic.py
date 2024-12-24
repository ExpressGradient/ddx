import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ddx import DDx



ddx = DDx(model='gpt-4o')

ddx.run("Nondimensionalize the polynomial\\[a_{1} x^{12} + a_{2} x^{8} + a_{3}\\]into one of the form $\\epsilon y^{12} + y^{8} + 1. $Express $\\epsilon$ as a function of $a_1$, $a_2$, and $a_3.$", verbose=True)