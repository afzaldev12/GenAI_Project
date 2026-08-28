from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_community.document_loaders import TextLoader,WebBaseLoader
import os



load_dotenv()

os.environ["USER_AGENT"] = "my-langchain-learning-project/1.0"

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


# Load model
model = load_llm()

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = "https://www.flipkart.com/search?q=apple%20macbook%20air%20m2&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'What is the product that we are talking aboout?', 'text':docs[0]}))