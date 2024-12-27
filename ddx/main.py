from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()


class DDx:
    def __init__(self, num_workers=5, model="gpt-4o"):
        self.n = num_workers
        self.model = model

        self.problem = ""
        self.__understanding = ""
        self.__representation = ""

        self.verbose = False

    def __chat(self, input_message: str, n=None) -> list[str]:
        if n is None:
            n = self.n

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "developer",
                    "content": "You are a highly intelligent assistant that can help in solving a complex problem in multiple phases",
                },
                {
                    "role": "user",
                    "content": input_message,
                },
            ],
            n=n,
        )

        return [choice.message.content for choice in response.choices]

    def __vprint(self, content: str):
        if self.verbose:
            print(content, "\n")

    def __understand_problem(self):
        self.__vprint("Generating problem understanding candidates...")

        candidates = self.__chat(f"""Goal: Grasp the following problem fully to set the foundation for a solution.

Sub-steps: 
Comprehend the Problem Statement
    - Break it down into smaller components
    - What is being asked? What does "solved" look like?
Identify Key Details
    - What information is provided?
    - What are the important variables, relationships or elements?
Clarify Ambiguities
    - Identify areas of vagueness or missing information
    - List down assumptions
Define Sucess Criteria:
    - Specify what constitutes a successful solution

Problem: {self.problem}""")

        self.__vprint("Merging the problem understanding candidates..")

        self.__understanding = self.__chat(
            f"""Goal: Merge the following problem understanding candidates into a single, comprehensive understanding.
The final understanding should be:
1. Complete: covers all important details of the problem
2. Clear: Easily to understand and logically structured.
3. Consistent: Resolves any conflicts or contradictions between the candidates.
4. Accurate: Aligns with the original problem statement.

Steps to follow:
1. Identify common elements across the candidates and include them in the final understanding.
2. Incorporate unique or complementary details from each candidate to enrich the final understanding.
3. Resolve any conflicting or ambiguous points by providing clear explanations or assumptions.
4. Present the final understanding in a structured and concise format.

Here are the problem understanding candidates: {"\n-----\n".join(candidates)}

Unified understanding: """,
            n=1,
        )[0]

        self.__vprint(f"Problem understanding: {self.__understanding}")

    def __represent_problem(self):
        self.__vprint("Generating problem representation candidates...")

        candidates = self.__chat(f"""Goal: Create a representation from the problem understanding.
Structure the problem into smaller, actionable components.

Follow these steps carefully:
1. Break down the problem:
    - Identify distinct subproblems or tasks required to solve the main problem.
    - Ensure that each subproblem is actionable, specific, and clear.

2. Define Relationships:
    - Determine how the subproblems relate to each other:
        - Sequential: One subproblem depends on the result of another.
        - Parallel: Subproblems can be solved independently.
        - Cyclic: Subproblems may involve iterative refinement.
    - Clearly describe these relationships.

3. List Constraints:
    - Identify any rules, limitations, or assumptions that must be satisfied.
    - Examples: Input constraints, boundary conditions, performance limits.

4. Organize the Representation:
    - Present your output in the following structure:
        - Subproblems
        - Relationships between subproblems
        - Constraints

Problem understanding: {self.__understanding}
""")

        self.__vprint("Merging problem representation candidates...")

        self.__representation = self.__chat(
            f"""Goal: Merge multiple problem representation candidates into a single, unified structure.
Follow these steps:
1. Analyze Individual Representations:
    - Review the provided problem representations
    - Identify commonalities across all representations
    - Highlight any unique or complementary elements in each representation.

2. Merge Subproblems:
    - Combine subproblems from each representation:
        - Include common subproblems.
        - Add unique or complementary subproblems to enrich the structure.
        - Remove redundant or overlapping subproblems.

3. Merge Relationships:
    - Consolidate relationships between subproblems:
        - Preserve dependencies
        - Resolve conflicts or inconsistencies.

4. Merge Constraints:
    - Combine the constraints from all representations.
    - Eliminate duplicates and clarify ambiguities.

5. Resolve Conflicts:
    - If there are contradictions or ambiguities, make assumptions or provide resolutions to ensure consistency.

6. Organize the Unified Representation:
    - Present the final output in the following structure:
        - Subproblems
        - Relationships between subproblems
        - Constraints

Problem representation candidates: {"\n-----\n".join(candidates)}

Unified representation: """,
            n=1,
        )[0]

        self.__vprint(f"Problem representation: {self.__representation}")

    def run(self, problem: str, verbose=False):
        self.problem = problem
        self.verbose = verbose

        self.__understand_problem()
        self.__represent_problem()
        return ""
