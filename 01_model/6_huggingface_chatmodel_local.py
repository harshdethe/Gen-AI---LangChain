from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os

# explicitally command det aho ki d drive madhe store kara 
os.environ['HF_HOME'] = 'D:/Data Sciences/Gen Ai'

# HuggingFacePipeline : Hee ek connector aahe jo LangChain la local (tumchya swatahchya computer/GPU var) 
# chalnarya Hugging Face model sobat jodto — internet var koni server la call na karta. 
# He transformers library cha pipeline() function vaprto, model directly RAM/GPU madhe load karto, 
# ani tithech inference (text generation) karto.
llm=HuggingFacePipeline.from_model_id(   #  ti tumchya sathi internally transformers cha pipeline तयार करते
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(  #He extra settings aahet je pratyaksha generation cha behavior control kartat
        temperature=0.5,
        max_new_tokens=100
    )
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("what is the capital of india")

print(result.content)