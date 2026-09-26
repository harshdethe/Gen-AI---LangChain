import os

# Force transformers to use the local cache only - no internet call needed
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# ---- CONFIG ----
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model... (one-time load, please wait)")

# Step 1: Create the raw transformers pipeline (same as before)
hf_pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device_map="auto",
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# Step 2: Wrap it in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipe)

# Step 3: Wrap THAT in ChatHuggingFace so it understands chat-style messages
# (TinyLlama-Chat is a chat-tuned model, so this gives proper role handling)
chat_model = ChatHuggingFace(llm=llm)

print("Model loaded!\n")

# ---- USAGE PATTERN 1: Direct invoke (like calling an API) ----
def ask_model_simple(prompt: str) -> str:
    """Simple one-off query using LangChain's invoke() - just like an API call."""
    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content


# ---- USAGE PATTERN 2: Using a PromptTemplate (LangChain-style) ----
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Tu ek helpful assistant ahes. Marathi ani English donhi madhe uttar deu shaktos."),
    ("user", "{question}"),
])

chain = prompt_template | chat_model  # this is LangChain's "chain" syntax


def ask_model_with_chain(question: str) -> str:
    """Query using a LangChain chain (prompt template -> model)."""
    response = chain.invoke({"question": question})
    return response.content


if __name__ == "__main__":
    # Example 1: simple invoke
    print("Q: Maharashtra chi rajdhani konti aahe?")
    print("A:", ask_model_simple("Maharashtra chi rajdhani konti aahe?"))
    print("-" * 50)

    # Example 2: using the chain with prompt template
    print("Q (via chain): Python madhe list ani tuple madhe farak kay?")
    print("A:", ask_model_with_chain("Python madhe list ani tuple madhe farak kay?"))
    print("-" * 50)

    # Interactive loop
    print("\nAta tu swatah prashna vicharu shaktos. Exit karaycha asel tar 'exit' lihi.\n")
    while True:
        user_input = input("Tu: ")
        if user_input.strip().lower() in ["exit", "quit", "q"]:
            print("Chat band kelay. Bye!")
            break
        reply = ask_model_with_chain(user_input)
        print("Model:", reply)
        print("-" * 50)