import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(os.path.join(os.path.dirname(__file__), "../", ".env"))

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL_NAME"),
)

prompt = input("Enter prompt:")

response = llm.invoke(prompt)

print(response.content)
