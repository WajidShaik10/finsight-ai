# quant_rag_agent/modules/agent.py

import os
import json
from groq import Groq
from quant_rag_agent.modules.retriever import DocumentRetriever

class QuantAgent:
    """
    RAG Agent with Smart Search + Memory.
    Step 1: LLM generates best search query
    Step 2: Search ChromaDB
    Step 3: LLM answers from results
    """

    def __init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.retriever = DocumentRetriever()
        self.model = "llama-3.3-70b-versatile"
        self.history = []
        print("Agent ready!")

    def generate_search_query(self, question):
        """
        Step 1: Ask LLM to rephrase the question
        into the best possible search query.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a search query generator.
Given a question, do one of two things:
1. If it's a math/calculation question → return "CALCULATE: <expression>" e.g. "CALCULATE: 25 * 4.5"
2. If it's a document question → return the best search query, nothing else.
No quotes, no explanation."""
                },
                {
                    "role": "user",
                    "content": f"Conversation history:\n{json.dumps(self.history[-4:], indent=2)}\n\nQuestion: {question}\n\nSearch query:"
                }
            ]
        )
        query = response.choices[0].message.content.strip()
        print(f"\n🔍 Search query: '{query}'")
        return query

    def calculate(self, expression):
        """Run a math calculation."""
        print(f"\n🔧 Calculating: {expression}")
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    def ask(self, question):
        """
        Agentic loop:
        1. Generate smart search query from question
        2. If math → calculate
        3. If document → search ChromaDB → answer
        """

        # Add user question to history
        self.history.append({
            "role": "user",
            "content": question
        })

        # Step 1: Generate smart search query
        search_query = self.generate_search_query(question)

        # Step 2: Check if it's a math question
        if search_query.startswith("CALCULATE:"):
            expression = search_query.replace("CALCULATE:", "").strip()
            result = self.calculate(expression)
            answer = f"The result of {expression} = {result}"

        else:
            # Step 3: Search ChromaDB
            chunks = self.retriever.retrieve(search_query, top_k=5)
            context = "\n\n".join(chunks)

            # Step 4: Answer from chunks
            system_prompt = f"""You are an expert financial analyst assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have that information."

CONTEXT:
{context}"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.history
                ]
            )
            answer = response.choices[0].message.content

        # Save to memory
        self.history.append({
            "role": "assistant",
            "content": answer
        })

        print(f"\n🤖 Agent: {answer}")
        return answer