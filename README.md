# DDx: Dynamic Diagnosis System for Mathematical Problem Solving

DDx (Dynamic Diagnosis) is an intelligent, iterative problem-solving system inspired by the critical-thinking process seen in *House M.D.*. It is designed to break down complex mathematical problems into manageable phases, leveraging multiple agents to collaboratively solve and refine solutions.

---

## 🚀 Features

- **Phased Problem Solving**: DDx splits problem-solving into six logical phases:
  1. **Understanding**: Decipher the problem statement and identify goals.
  2. **Decomposition**: Break the problem into smaller, solvable components.
  3. **Planning**: Create a strategy to solve the subproblems.
  4. **Execution**: Solve each component systematically.
  5. **Verification**: Validate the solutions against the original problem.
  6. **Compilation**: Combine results into a cohesive final answer.

- **Interactive Agents**: Mimics the dynamic interaction of "House" (critical guide) and "Team" (problem solvers) to iteratively improve solutions.

- **Tool Integration**: Leverages symbolic computation (e.g., SymPy) for mathematical operations during the execution phase.

- **Verbose Debugging**: Option to enable detailed output for tracking agent reasoning and decision-making.

---

## 🧠 How It Works

1. **Input a Problem**: Provide a complex mathematical or symbolic problem to DDx.
2. **Iterative Problem-Solving**:
   - The "House" agent oversees the process, iteratively questioning and refining the solution.
   - The "Team" agent generates outputs for each phase.
3. **Tools & Execution**: Utilize computational tools like SymPy for precise calculations.
4. **Verification & Refinement**: Validate and refine solutions until the "House" agent is satisfied.
5. **Final Answer**: Output a fully solved and explained solution.

---

## 🔨 How to use?

```python
from ddx_ai import DDx

DDx("<question>", verbose=False)
```
---

## 📄 Example Usage

### Input Problem
Nondimensionalize the polynomial:
\[
P(x) = a_1 x^{25} + a_2 x + a_3
\]
into the form:
\[
\epsilon y^{25} + y + 1
\]
Express \(\epsilon\) as a function of \(a_1\), \(a_2\), and \(a_3\).

### Output
1. **Understanding Phase**:
   - Identify the goal: Transform the polynomial into the specified nondimensional form.
   - Extract key coefficients: \(a_1\), \(a_2\), \(a_3\).

2. **Decomposition Phase**:
   - Break the problem into:
     - Variable scaling (\(L\)) to make \(y^1\) coefficient equal to 1.
     - Normalizing the constant term to 1.
     - Calculating \(\epsilon\).

3. **Execution Phase**:
   - Perform substitutions: \(x = L y\), where \(L = \frac{1}{a_2}\).
   - Solve for \(\epsilon = \frac{a_1 a_3^{24}}{a_2^{25}}\).

4. **Verification Phase**:
   - Substitute back into the polynomial to ensure it matches the desired form.

5. **Compilation Phase**:
   - Final answer:
     \[
     \boxed{\epsilon = \frac{a_1 a_3^{24}}{a_2^{25}}}
     \]

---

## 💡 Key Design Principles

1. **Iterative Refinement**:
   - Mimics real-world diagnostic processes to refine solutions through critical questioning.
2. **Dynamic Interactions**:
   - Encourages agents to think critically, ensuring high-quality solutions.
3. **Phased Approach**:
   - Logical segmentation for tackling complex problems methodically.

