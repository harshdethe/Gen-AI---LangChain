from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)


# Low temperature (0–0.3) = जास्त अचूक, कमी randomness.
# Medium (0.5–1.0) = अचूकता + थोडी creativity.
# High (1.2–2.0) = जास्त creativity, जास्त randomness.
# ex 
# Temperature = 0 → शिक्षक जसा पुस्तकातलं नेमकं उत्तर देतो.
# Temperature = 0.8 → शिक्षक पुस्तकासोबत स्वतःची उदाहरणंही देतो.
# Temperature = 1.8 → लेखक किंवा storyteller सारखा भरपूर कल्पना वापरून उत्तर देतो.