import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv(os.path.join(os.path.dirname(__file__), "../", ".env"))

model = ChatGroq(
    model_name=os.getenv("GROQ_MODEL_NAME"),
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# Calculator Agent
@tool
def square_number(number: int) -> str:
    """Returns the square of a number."""
    result = number * number
    return f"The square of {number} is {result}."

calculator_sys_msg = SystemMessage(
    content="You are a Math Calculator. Use your tool to compute numbers."
)

calculator = create_agent(
    model,
    tools=[square_number],
    system_prompt=calculator_sys_msg
)

# --- Explainer Agent ---
explainer_sys_msg = SystemMessage(
    content="You are a teacher. Explain mathematical results in a simple way."
)

explainer = create_agent(
    model,
    system_prompt=explainer_sys_msg
)

# --- Orchestration using HumanMessage, AIMessage, and SystemMessage ---

number_to_calculate = 7
print(f"Calculating square for: {number_to_calculate}\n")

# 1. Calculator computes the square
calc_query = f"Find the square of {number_to_calculate}"

calc_response = calculator.invoke({
    "messages": [HumanMessage(content=calc_query)]
})

result = calc_response["messages"][-1].content
print(f"Calculator Output: {result}\n")


# 2. Explainer receives result as conversation history
explainer_history = [
    AIMessage(content=result),
    HumanMessage(content="Explain this result in a simple way for a beginner.")
]

explainer_response = explainer.invoke({
    "messages": explainer_history
})

explanation = explainer_response["messages"][-1].content
print(f"Explainer Output:\n{explanation}")
