from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()



def load_llm():
    hf_pipeline = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 256,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "return_full_text": False,
        },
    )
    return ChatHuggingFace(llm=hf_pipeline)


model = load_llm()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Unemployment in Pakistan'})

print(result)

chain.get_graph().print_ascii()
