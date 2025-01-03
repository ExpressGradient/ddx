from openai import OpenAI

client = OpenAI()


class Phase:
    def __init__(self, name, goal):
        self.name = name
        self.goal = goal
        self.is_completed = False


phases = [
    Phase(
        "Understanding",
        """- Clearly define the problem statement.
- Identify and list key constraints and parameters of the problem.
- Highlight known information and assumptions.
- Ensure alignment on what success looks like.""",
    ),
    Phase(
        "Decomposition",
        """- Break the problem into smaller, manageable components or subproblems.
- Define the relationships and dependencies between components.
- Identify areas requiring further exploration or clarification.
- Prioritize components based on importance or urgency""",
    ),
    Phase(
        "Planning",
        """- Develop a clear strategy or sequence of steps to address each component.
- Assign methods, tools, or techniques for solving each subproblem,
ensuring compatibility with SymPy as the only computational tool available during execution.
- Identify how Sympy's capabilities will be used to solve specific subproblems.
- Identify potential challenges and risks in the plan.
- Outline checkpoints or milestones to track progress.""",
    ),
    Phase(
        "Execution",
        """- Implement the plan by solving the defined subproblems.
- Generate solutions for each component, ensuring logical consistency.
- Document the process and intermediate results for transparency.
- Maintain adaptability to adjust the plan if needed.""",
    ),
    Phase(
        "Verification",
        """- Validate the results against the problem constraints and the success criteria.
- Perform error checking and ensure logical soundness of the solution.
- Identify and inconsistencies, gaps, or areas needing revision.
- Confirm that the solution integrates seamlessly across components.""",
    ),
    Phase(
        "Compilation",
        """- Synthesize the verified solution into a single, coherent final answer.
- Provide a clear and concise explanation of the solution, including the reasoning and steps involved.
- Format the result to ensure it is actionable and easily understood by the intended audience.
- Ensure completeness and clarity, highlighting the problem's resolution and any relevant thoughts.""",
    ),
]
