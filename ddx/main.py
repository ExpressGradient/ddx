from openai import OpenAI

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

    def _initialize_understanding(self, question):
        return self._openai_ask(
            f"Analyze the following question and generate your understanding of it.\nQuestion: {question}"
        )


class DDX:
    def __init__(self, num_agents=3, model="gpt-4o-mini"):
        self.agents = [Agent(f"agent-{i}", model) for i in range(num_agents)]
        self.understanding_artifact = ""

        self.verbose = False

    def vprint(self, message):
        if self.verbose:
            print(message)

    def _create_understanding(self, question):
        self.vprint("Understanding the question...")
        self.understanding_artifact = self.agents[0]._initialize_understanding(question)

    def run(self, question):
        self._create_understanding(question)
