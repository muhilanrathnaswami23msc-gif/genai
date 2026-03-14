import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv(os.path.join(os.path.dirname(__file__), "../", ".env"))

# Initialize model
model = ChatGroq(
    model_name=os.getenv("GROQ_MODEL_NAME"),
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# --- Healthcare Tools ---

@tool
def symptom_checker(symptom: str) -> str:
    """Provides basic medical guidance."""
    return f"For symptoms related to '{symptom}', please consult a doctor."

@tool
def appointment_scheduler(name: str, phone: str) -> str:
    """Schedules a doctor appointment."""
    return f"Appointment request received for {name}. We will contact you at {phone}."


# --- Agent with Custom PII Middleware ---

agent = create_agent(
    model=model,
    tools=[symptom_checker, appointment_scheduler],
    middleware=[

        # Detect names like "John Smith"
        PIIMiddleware(
            "patient_name",
            detector=r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",
            strategy="redact",
            apply_to_input=True,
        ),

        # Detect 10-digit phone numbers
        PIIMiddleware(
            "phone_number",
            detector=r"\b\d{10}\b",
            strategy="mask",
            apply_to_input=True,
        ),

        # Detect insurance IDs like INS-12345678
        PIIMiddleware(
            "insurance_id",
            detector=r"INS-\d{8}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)


# --- Test Input ---

content1 = "Hello, my name is John Smith. My phone number is 9876543210."
content2 = "My insurance ID is INS-12345678."

print(f"Original Input:\n{content1}\n")

try:
    result = agent.invoke({
        "messages": [HumanMessage(content=content1)]
    })

    processed_input = result["messages"][0].content
    print(f"Input as seen by Model (after Middleware):\n{processed_input}\n")

    final_answer = result["messages"][-1].content
    print(f"Agent Final Answer:\n{final_answer}")

    # Test input with blocked insurance ID

    result = agent.invoke({
        "messages": [HumanMessage(content=content2)]
    })

    processed_input = result["messages"][0].content
    print(f"Input as seen by Model (after Middleware):\n{processed_input}\n")

    final_answer = result["messages"][-1].content
    print(f"Agent Final Answer:\n{final_answer}")

except Exception as e:
    print(f"Agent blocked input:\n{e}")
