from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# low level connector jo hugging face inference API la jodte
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task='text-generation'
)

# हा त्या वरच्या HuggingFaceEndpoint ला wrap करतो आणि त्याला chat-compatible बनवतो.
model=ChatHuggingFace(llm=llm)

result=model.invoke("what is the capital of india")

print(result.content)

