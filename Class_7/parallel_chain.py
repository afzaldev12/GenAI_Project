from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
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


model1 = load_llm()
model2 = load_llm()


prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Marge the provided notes and quiz into a single document \n note -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 |parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """Artificial Intelligence (AI) is transforming industries across the world by enabling machines to perform tasks that typically require human intelligence. AI technologies such as machine learning, natural language processing, and computer vision are being used in healthcare, finance, education, manufacturing, and transportation.

In the healthcare sector, AI assists doctors by analyzing medical images, predicting diseases, and recommending treatment plans. Hospitals also use AI-powered chatbots to answer patient questions and schedule appointments. These applications improve efficiency and reduce the workload of healthcare professionals.

The financial industry relies on AI for fraud detection, credit scoring, algorithmic trading, and customer support. Machine learning models analyze millions of transactions in real time to identify suspicious activities, helping banks prevent financial losses.

Educational institutions use AI to create personalized learning experiences. Intelligent tutoring systems analyze student performance and recommend learning resources based on individual strengths and weaknesses. Automated grading systems also help teachers save time.

Despite its benefits, AI raises ethical concerns such as data privacy, algorithmic bias, and job displacement. Organizations must ensure that AI systems are transparent, fair, and secure. Governments and technology companies are working together to establish regulations and ethical guidelines for responsible AI development.

As AI continues to evolve, professionals must develop new skills to work effectively alongside intelligent systems. Continuous learning and responsible innovation will play a key role in shaping the future of AI and its impact on society.
"""

result = chain.invoke({'text':text}) 

print(result)

chain.get_graph().print_ascii()