from openai import OpenAI
from pydantic import BaseModel, Field

client = OpenAI()


class Agent:
    def __init__(self, id, model):
        self.id = id
        self.model = model

        self.understanding_consensus = False
        self.plan = []
        self.solution = ""

    def _openai_ask(self, query):
        return (
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent agent participating in a group to solve a complex problem",
                    },
                    {"role": "user", "content": query},
                ],
            )
            .choices[0]
            .message.content
        )

    def _openai_parse(self, query, response_format: BaseModel):
        return (
            client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent agent participating in a group to solve a complex problem",
                    },
                    {"role": "user", "content": query},
                ],
                response_format=response_format,
            )
            .choices[0]
            .message.parsed
        )

    def _initialize_understanding(self, question):
        return self._openai_ask(
            f"""Generate your initial understandings of the following question
            Question: {question}"""
        )

    def _analyze_understanding(self, question, understanding):
        class UnderstandingConsensus(BaseModel):
            is_satisfied_with_understanding: bool = Field(
                ...,
                description="Is satisified with the understanding of the question or not",
            )
            feedback: str = Field(
                ...,
                description="Feedback of the understanding of the question if not satisfied",
            )

        understanding_analysis_response = self._openai_parse(
            f"""Critique the following understanding of the question.
            It should cover the full picture of the question.
            It shouldn't miss any important details.
            Try looking at the question from different angles.
            If not satisified with it, provide feedback on how it can be improved.
            Reminder: Goal is to understand the question as best as possible.
            Question: {question}
            Understanding: {understanding}""",
            UnderstandingConsensus,
        )

        return understanding_analysis_response

    def _update_understanding_with_feedback(self, understanding, feedback):
        return self._openai_ask(
            f"""Update the following understanding based on the feedback.
            Understanding: {understanding}
            Feedback: {feedback}"""
        )


class DDx:
    def __init__(self, num_agents=3, model="gpt-4o-mini"):
        self.agents = [Agent(f"agent-{i}", model) for i in range(num_agents)]
        
        self.question = ""
        self.understanding_artifact = ""
        
        self.max_understanding_iterations = 5

        self.verbose = False

    def _vprint(self, message):
        if self.verbose:
            print(message, "\n")

    def _create_understanding(self):
        self._vprint("Understanding the question...")
        self.understanding_artifact = self.agents[0]._initialize_understanding(
            self.question
        )
        self._vprint(f"Initial understanding: {self.understanding_artifact}")

    def _critique_understanding(self):
        self._vprint("Critiquing the understanding...")

        iteration = 0

        while not all([agent.understanding_consensus for agent in self.agents]):
            for agent in self.agents:
                if not agent.understanding_consensus:
                    critique = agent._analyze_understanding(
                        self.question, self.understanding_artifact
                    )

                    if critique.is_satisfied_with_understanding:
                        agent.understanding_consensus = True
                        self._vprint(f"{agent.id} is satisfied with the understanding.")
                    else:
                        self.understanding_artifact = (
                            agent._update_understanding_with_feedback(
                                self.understanding_artifact, critique.feedback
                            )
                        )
                        self._vprint(
                            f"{agent.id} is not satisfied with the understanding."
                        )
                        self._vprint(f"Feedback: {critique.feedback}")

            iteration += 1

            if iteration >= self.max_understanding_iterations:
                self._vprint("Max understanding iterations reached.")
                break

        self._vprint("Understanding consensus reached.")
        self._vprint(f"Final understanding: {self.understanding_artifact}")

    def run(self, question, max_understanding_iterations=5, verbose=False):
        self.question = question

        self.max_understanding_iterations = max_understanding_iterations
        self.verbose = verbose

        self._create_understanding()
        self._critique_understanding()
