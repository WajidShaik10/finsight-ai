# main.py
import os
from quant_rag_agent.modules.ingester import DocumentIngester
from quant_rag_agent.modules.agent import QuantAgent

from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()

# Ingest Apple 10-K
ingester = DocumentIngester()
ingester.ingest("quant_rag_agent/data/apple_10k_2023.html")

# Chat with agent
agent = QuantAgent()

def main():
    print("\n🤖 Quant RAG Agent — Apple 10-K loaded!")
    print("📚 Ask me anything about Apple's financials")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! 👋")
            break
        if not user_input:
            continue
        agent.ask(user_input)

if __name__ == "__main__":
    main()