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
        self.__solution_plan = []

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

    def __parse(self, query: str, response_format: BaseModel):
        return (
            client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "developer",
                        "content": "You are a highly intelligent assistant that can help in solving a complex problem in multiple phases",
                    },
                    {"role": "user", "content": query},
                ],
                response_format=response_format,
            )
            .choices[0]
            .message.parsed
        )

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
1. Analyze the problem understanding and split into smaller logical subproblems.
2. Arrange the subproblems in the order they should be solved.
3. Ensure the sequence is logical and follows any natural dependencies (e.g., validate inputs before computation)
4. Identify and constraints that must be satisfied for the solution.

Present your output in the following structure:
1. Subproblems (listed in sequential order)
2. Constraints.

Problem understanding: {self.__understanding}

Generate a structured problem representation: 
""")

        self.__vprint("Merging problem representation candidates...")

        self.__representation = self.__chat(
            f"""Goal: Merge multiple problem representation candidates into a single, unified structure.
Follow these steps:

1. Review the provided problem representations.
2. Identify common subproblems across all representations.
3. Include unique or complementary subproblems to enrich the final structure.
4. Arrange the subproblems in a logical and sequential order.
5. Merge the constraints and resources from all representations, ensuring no duplicates or contradictions.
6. Present the final output in the following structure:
    - Subproblems (sequential order)
    - Constraints

Problem representation candidates: {"\n-----\n".join(candidates)}

Unified representation: """,
            n=1,
        )[0]

        self.__vprint(f"Problem representation: {self.__representation}")

    def __plan_solution(self):
        self.__vprint("Generating solution planning candidates...")

        candidates = self.__chat(f"""Goal: Generate a solution plan to a complex problem given its subproblems and constraints.
Follow these steps:
1. Analyze the list of subproblems. 
2. For each subproblem, propose:
    - A clear and actionable method to solve it.
    - The required inputs for the method.
    - The expected outputs after execution.
3. Arrange these methods into a sequential plan.
4. Ensure the plan adheres to the given constraints.

Problem representation: {self.__representation}

Generate the complete solution plan:""")

        self.__vprint("Merging solution planning candidates...")

        solution_plan_str = self.__chat(
            f"""Goal: Merge the multiple proposed solution plans into a single, unified plan.
Follow these steps:

1. Compare the proposed plans and identify commonalities.
2. Incorporate unique or complementary elements from each plan.
3. Resolve any conflicts or contradictions.
4. Ensure it addresses all subproblems in sequential order.
5. Present the final unified plan in a clear, step-by-step format.
6. Verify that inputs and outputs are clearly defined for each step.

Proposed plans: {candidates}

Generate the unified solution plan:""",
            n=1,
        )[0]

        self.__vprint("Extracting contents from solution plan...")

        class Step(BaseModel):
            description: str
            method: str
            input: str
            expected_output: str

        class SolutionPlan(BaseModel):
            steps: list[Step] = Field(
                ...,
                description="List of steps (with description, method, inputs and outputs) in the solution plan",
            )
            constraints: str = Field(
                ..., description="Constraints mentioned in the solution plan"
            )

        self.__solution_plan = self.__parse(
            f"""Extract the list of steps and constraints from the solution plan: {solution_plan_str}""",
            SolutionPlan,
        )

        self.__vprint(f"Solution plan: {self.__solution_plan}")

    def run(self, problem: str, verbose=False):
        self.problem = problem
        self.verbose = verbose

        self.__understand_problem()
        self.__represent_problem()
        self.__plan_solution()
        return ""
