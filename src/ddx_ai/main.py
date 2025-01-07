import inspect
import io
import json
from contextlib import redirect_stdout

import sympy
from openai import OpenAI
from pydantic import BaseModel, Field
from sympy import Eq, solve, symbols

client = OpenAI()


class Phase:
    def __init__(self, name, goals):
        self.name = name
        self.goals = goals
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


def generate_openai_tool_spec(functions):
    if functions is None:
        return None

    tool_specs = []

    for func in functions:
        name = func.__name__
        description = func.__doc__
        parameters = {"type": "object", "properties": {}, "required": []}

        signature = inspect.signature(func)

        for param_name, param in signature.parameters.items():
            param_type = (
                "string"
                if param.annotation in [str, inspect.Parameter.empty]
                else "number"
                if param.annotation in [int, float]
                else "object"
            )

            parameters["properties"][param_name] = {"type": param_type}
            if param.default is not inspect.Parameter.empty:
                parameters["required"].append(param_name)

        tool_spec = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }

        tool_specs.append({"type": "function", "function": tool_spec})

    return tool_specs


class Agent:
    def __init__(self, name, developer_message, tools=None, verbose=False):
        self.name = name
        self.messages = [{"role": "developer", "content": developer_message}]
        self.tools = tools
        self.tool_specs = generate_openai_tool_spec(self.tools)

        self.verbose = verbose

    def vprint(self):
        if self.verbose:
            print(f"FROM {self.name}: {self.messages[-1]['content']}\n")

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        self.vprint()

        if self.tool_specs is None:
            response = client.chat.completions.create(
                model="gpt-4o", messages=self.messages
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages,
                tools=self.tool_specs,
                tool_choice="auto",
            )

        self.messages.append(dict(response.choices[0].message))
        self.vprint()

        if response.choices[0].finish_reason == "tool_calls":
            for call_dict in response.choices[0].message.tool_calls:
                tool_name = call_dict.function.name
                tool_id = call_dict.id
                tool_args = json.loads(call_dict.function.arguments)

                tool_to_call = [t for t in self.tools if t.__name__ == tool_name][0]
                result = tool_to_call(**tool_args)

                tool_message = {
                    "role": "tool",
                    "content": json.dumps({**tool_args, "result": result}),
                    "tool_call_id": tool_id,
                }
                self.messages.append(tool_message)
                self.vprint()

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages,
            )

            self.messages.append(dict(response.choices[0].message))
            self.vprint()

        return self.messages[-1]["content"]

    def check_phase_completion(self):
        class PhaseCompletion(BaseModel):
            is_phase_completed: bool = Field(
                ..., description="Is this phase completed?"
            )

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=self.messages
            + [
                {
                    "role": "user",
                    "content": """Did the team satisfy all the goals for the current phase?
All goals, not just a few. Also did the team fully complete the steps it laid?
Is it the time to move on to the next phase.""",
                }
            ],
            response_format=PhaseCompletion,
        )

        return response.choices[0].message.parsed.is_phase_completed


house = Agent(
    "House",
    """You are "House," the guiding and oversight agent in the DDx problem-solving system. Your role is to:
- Lead the process by systematically progressing through the phases (Understanding, Decomposition, Planning, Execution, Verification, Compilation).
- Stay within each phase and continue to provide targeted feedback, ask questions, and request refinements until fully satisfied with the quality of the output.
- Demand thorough and precise responses from the "Team" agent, ensuring that each phase is rigorously completed before progressing to the next.
- Request revisions, clarifications, or refinements as needed to ensure the solution aligns with the problem's requirements.
- Maintain a high-level perspective, ensuring all phases are connected and aligned toward the end goal.

**Tone and Style:**
- Be critical, inquisitive, and assertive, but constructive.
- Push the "Team" agent to think deeply and provide detailed, high-quality responses.
- Ensure the problem-solving process remains rigorous, focused, and iterative, improving outputs through repeated interaction.""",
)


def exec_sympy_code(code: str, allowed_modules: dict = None):
    """
    Safely execute SymPy code generated by an agent.

    The agent must ensure that:

    **Agent Instructions**:
    - Your code must `print()` the final result explicitly.
    - Intermediate calculations are allowed but should also use `print()` for visibility if necessary.
    - Always ensure that the result is clearly labeled in the printed output for easy identification.


    **Allowed Features**:
    - Your code can use all SymPy operations for symbolic computation.
    - Symbols must be explicitly declared using `symbols()`.
    - Use `print()` to output any intermediate steps and the final result.

    Args:
        code (str): The SymPy code to evaluate.
                    The code must include `print()` statements for results.
        allowed_modules (dict): Allowed modules and variables for execution.

    Returns:
        str: String of the result of the execution.
    """
    if allowed_modules is None:
        allowed_modules = {"sympy": sympy, "symbols": symbols, "Eq": Eq, "solve": solve}

    local_env = {}
    output_buffer = io.StringIO()

    with redirect_stdout(output_buffer):
        exec(code, {"__builtins__": __builtins__, **allowed_modules}, local_env)

    output = output_buffer.getvalue().strip()
    return str(output)


team = Agent(
    "Team",
    """You are "Team," the executor and generator agent in the DDx problem-solving system. Your role is to:
- Respond to "House" by generating outputs for each requested phase (Understanding, Decomposition, Planning, Execution, Verification, Compilation).
- Provide detailed, thoughtful, and structured responses that address the problem requirements.
- Adapt and refine your outputs based on "House's" feedback, iterating as needed.
- Utilize SymPy as your sole computational tool for solving mathematical problems during the Execution phase and where relevant in other phases.
- Maintain a problem-focused approach, ensuring all responses are actionable and aligned with the end goal.

**Tone and Style:**
- Be clear, concise, and professional.
- Provide evidence, reasoning, or supporting details for your outputs.
- Collaborate effectively by responding to "House's" inquiries and requests constructively.

**Tool Access:**
- You have access to SymPy, a symbolic mathematics library, for tasks involving algebra, calculus, equation solving, and other symbolic computations.
- Ensure all computational solutions are generated using SymPy, and clearly explain the methods and results.
- Ensure that you always print the final result in the execution.
- Try not calling the tool as soon as you receive the problem. Go through phases.""",
    tools=[exec_sympy_code],
)


def DDx(question, verbose=False):
    print("Initializing DDx...")

    house.verbose = verbose
    team.verbose = verbose

    house.chat(
        f"The user has posed the following question: {question}. Let's start working on this."
    )
    team.chat(f"House got a new question for us to work on: {question}")

    for phase in phases:
        print(f"Starting {phase.name} phase")

        house_response = house.chat(
            f"The phase is {phase.name} and its goals are {phase.goals}. What should the team do?"
        )

        while not phase.is_completed:
            team_response = team.chat(f"House asks for: {house_response}")
            house_response = house.chat(f'Team responded with: "{team_response}"')

            if house.check_phase_completion():
                phase.is_completed = True

    return team.chat("The final answer is: ")
