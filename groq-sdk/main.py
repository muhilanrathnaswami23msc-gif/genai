import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../", ".env"))

api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL_NAME")

client = Groq(
    api_key=api_key
)

message = input("Enter message:")

completion = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": message,
        }
    ],
)

print(completion.choices[0].message.content)
