import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ddx import DDx


ddx = DDx(model="gpt-4o")

ddx.run(
    "Consider the polynomial\\[P(x) =\\epsilon x^{8} + x^{2} - 1.\\] \nFind first order approximations for all roots of the polynomial in the limit of small positive $\\epsilon$ and large positive $\\epsilon$.",
    verbose=True,
)
