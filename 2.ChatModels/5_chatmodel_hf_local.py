from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

import os



os.environ["HF_HOME"] = 'D:/My Projects/Langchain_models/.cache/huggingface'

llm = HuggingFacePipeline.from_model-id(

    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",

    task="text-generation",

    model_kwargs={"temperature": 0.9, "max_new_tokens": 1000}

)



model = ChatHuggingFace(llm=llm)



result = model.invoke("What is the name of the prime minister of Pakistan?")



print(result.content)