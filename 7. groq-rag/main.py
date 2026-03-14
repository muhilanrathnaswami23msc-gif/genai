import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../", ".env"))

def run_rag_demo():

    # 0. Configuration
    index_path = "faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 1. Check if index exists
    if os.path.exists(index_path):
        print(f"✓ Loading existing FAISS index from {index_path}...")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    else:
        print("! FAISS index not found. Creating new one...")

        # 1a. Knowledge base about Agentic AI
        text_data = """
        Agentic AI refers to artificial intelligence systems that can act autonomously
        to achieve goals. These systems are capable of planning tasks, making decisions,
        using tools, and adapting their behavior based on feedback.

        An AI agent is a software entity powered by an AI model that can perceive inputs,
        reason about them, and take actions to accomplish a task. AI agents often use
        large language models to understand instructions and generate responses.

        Agentic AI systems are commonly used in automation, research assistance,
        software development, customer support, and data analysis. They can break
        complex problems into smaller steps and execute them sequentially.

        The difference between a simple AI agent and an Agentic AI system is mainly
        the level of autonomy. A basic AI agent usually performs a single task when
        prompted, such as answering a question or summarizing text. In contrast,
        Agentic AI systems can plan multi-step workflows, coordinate tools,
        interact with external systems, and operate with minimal human guidance.

        For example, an AI coding agent might search documentation, generate code,
        run tests, and debug errors automatically. This ability to plan, execute,
        and adapt makes Agentic AI a powerful approach for building intelligent
        automation systems.
        """

        # 1b. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )

        docs = text_splitter.create_documents([text_data])
        print(f"✓ Created {len(docs)} chunks from text.")

        # 1c. Create vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(index_path)

        print(f"✓ FAISS vector store created and saved to {index_path}.")

    # 2. Initialize LLM
    llm = ChatGroq(
        model_name=os.getenv("GROQ_MODEL_NAME"),
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    # 3. Setup RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # 4. Questions related to Agentic AI

    query1 = "What is Agentic AI and what is it used for?"
    print(f"\nQuestion: {query1}")

    response1 = qa_chain.invoke(query1)
    print(f"\nAnswer: {response1['result']}")

    query2 = "What is an AI agent and how does it work?"
    print(f"\nQuestion: {query2}")

    response2 = qa_chain.invoke(query2)
    print(f"\nAnswer: {response2['result']}")

    query3 = "What is the difference between an AI agent and Agentic AI systems?"
    print(f"\nQuestion: {query3}")

    response3 = qa_chain.invoke(query3)
    print(f"\nAnswer: {response3['result']}")


if __name__ == "__main__":
    run_rag_demo()
