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

# --- Calculator Agent ---
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

# --- Streaming Helper Function (same as your first code) ---
def stream_agent(agent, inputs, agent_name):
    print(f"--- {agent_name} Streaming Start ---")
    final_content = ""

    for chunk in agent.stream(inputs, stream_mode="updates"):
        for node, data in chunk.items():
            print(f"\n[Node: {node}]")

            if "messages" in data:
                msg = data["messages"][-1]

                if msg.content:
                    print(f"Content: {msg.content}")
                    final_content = msg.content

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"Tool Calls: {msg.tool_calls}")

    print(f"\n--- {agent_name} Streaming End ---\n")
    return final_content


# --- Orchestration with Streaming ---

number_to_calculate = 7
print(f"Calculating square for: {number_to_calculate}\n")

# 1. Calculator finds the square (Streaming)

calc_inputs = {
    "messages": [
        HumanMessage(content=f"Find the square of {number_to_calculate}")
    ]
}

result = stream_agent(calculator, calc_inputs, "Calculator")


# 2. Explainer receives result (Streaming)

explainer_inputs = {
    "messages": [
        AIMessage(content=result),
        HumanMessage(content="Explain this result in a simple way for a beginner.")
    ]
}

explanation = stream_agent(explainer, explainer_inputs, "Explainer")


print("Final Explanation:\n")
print(explanation)
