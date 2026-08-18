from abc import ABC, abstractmethod


# ============================================================
# 1. Base Runnable Class
# ============================================================

class Runnable(ABC):

    @abstractmethod
    def invoke(self, input_data):
        pass


# ============================================================
# 2. Custom Fake LLM
# ============================================================

class NakliLLM(Runnable):

    def __init__(self):
        print("LLM Created")

    def invoke(self, prompt):

        print(f"\nLLM received prompt: {prompt}")

        prompt_lower = prompt.lower()

        # ----------------------------------------------------
        # Question: Capital of Pakistan
        # ----------------------------------------------------
        if "capital of pakistan" in prompt_lower:

            return {
                "response": "Islamabad is the capital of Pakistan."
            }

        # ----------------------------------------------------
        # Write a joke about Cricket
        # ----------------------------------------------------
        elif "write a joke about cricket" in prompt_lower:

            return {
                "response": (
                    "Why did the cricket player bring a ladder to the match? "
                    "Because he wanted to reach the next level!"
                )
            }

        # ----------------------------------------------------
        # Explain the joke
        # ----------------------------------------------------
        elif "explain the following joke" in prompt_lower:

            return {
                "response": (
                    "The joke uses the phrase 'reach the next level' in two ways. "
                    "A ladder helps someone physically climb higher, while in sports "
                    "and games, reaching the next level means improving or progressing."
                )
            }

        # ----------------------------------------------------
        # Write a poem about Pakistan
        # ----------------------------------------------------
        elif "poem" in prompt_lower and "pakistan" in prompt_lower:

            return {
                "response": (
                    "Pakistan, a land of mountains high,\n"
                    "Where rivers flow beneath the sky.\n"
                    "From green valleys to deserts wide,\n"
                    "A nation filled with hope and pride."
                )
            }

        # ----------------------------------------------------
        # Default response
        # ----------------------------------------------------
        else:

            return {
                "response": "Sorry, I do not have a response for this prompt."
            }


    def predict(self, prompt):
        return self.invoke(prompt)


# ============================================================
# 3. Custom Prompt Template
# ============================================================

class NakliPromptTemplate(Runnable):

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_data):

        return self.template.format(**input_data)

    def format(self, input_dict):

        return self.template.format(**input_dict)


# ============================================================
# 4. Custom String Output Parser
# ============================================================

class NakliStrOutputParser(Runnable):

    def invoke(self, input_data):

        return input_data["response"]


# ============================================================
# 5. Runnable Connector
# ============================================================

class RunnableConnector(Runnable):

    def __init__(self, runnable_list):

        self.runnable_list = runnable_list

    def invoke(self, input_data):

        for runnable in self.runnable_list:

            input_data = runnable.invoke(input_data)

        return input_data


# ============================================================
# EXAMPLE 1: Simple LLM
# ============================================================

print("\n========== EXAMPLE 1: SIMPLE LLM ==========")

llm = NakliLLM()

result = llm.predict("What is the capital of Pakistan?")

print("Result:", result)


# ============================================================
# EXAMPLE 2: Prompt Template
# ============================================================

print("\n========== EXAMPLE 2: PROMPT TEMPLATE ==========")

template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

prompt = template.invoke({
    "length": "short",
    "topic": "Pakistan"
})

print("Formatted Prompt:", prompt)


# ============================================================
# EXAMPLE 3: Simple Chain
# Prompt -> LLM
# ============================================================

print("\n========== EXAMPLE 3: SIMPLE CHAIN ==========")

llm = NakliLLM()

chain = RunnableConnector([
    template,
    llm
])

result = chain.invoke({
    "length": "short",
    "topic": "Pakistan"
})

print("Chain Result:", result)


# ============================================================
# EXAMPLE 4: Complete Chain
# Prompt -> LLM -> Parser
# ============================================================

print("\n========== EXAMPLE 4: COMPLETE CHAIN ==========")

template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

llm = NakliLLM()

parser = NakliStrOutputParser()

chain = RunnableConnector([
    template,
    llm,
    parser
])

result = chain.invoke({
    "length": "long",
    "topic": "Pakistan"
})

print("Final Result:", result)


# ============================================================
# EXAMPLE 5: FINAL CHAIN
# chain1 -> chain2
# ============================================================

print("\n========== EXAMPLE 5: FINAL CHAIN ==========")


template1 = NakliPromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)


template2 = NakliPromptTemplate(
    template="Explain the following joke: {response}",
    input_variables=["response"]
)


llm = NakliLLM()

parser = NakliStrOutputParser()


# Chain 1: Prompt -> LLM
chain1 = RunnableConnector([
    template1,
    llm
])


# Chain 2: Prompt -> LLM -> Parser
chain2 = RunnableConnector([
    template2,
    llm,
    parser
])


# Final Chain: Chain1 -> Chain2
final_chain = RunnableConnector([
    chain1,
    chain2
])


result = final_chain.invoke({
    "topic": "Cricket"
})


print("\nFinal Output:")
print(result)