# quant_rag_agent/modules/agent.py

import os
import json
from groq import Groq
from serpapi import GoogleSearch
from quant_rag_agent.modules.retriever import DocumentRetriever

class QuantAgent:
    def __init__(self, collection_name="quant_documents"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.retriever = DocumentRetriever(collection_name=collection_name)
        self.model = "llama-3.3-70b-versatile"
        self.history = []
        self.serpapi_key = os.environ.get("SERPAPI_KEY")
        print("Agent ready!")

    # ─────────────────────────────────────────
    # TOOLS
    # ─────────────────────────────────────────

    def search_documents(self, query):
        """Search ChromaDB for relevant chunks."""
        print(f"\n📄 Searching documents: '{query}'")
        chunks = self.retriever.retrieve(query, top_k=5)
        return "\n\n".join(chunks)

    def search_web(self, query):
        """Search the internet for live data."""
        print(f"\n🌐 Searching web: '{query}'")
        try:
            search = GoogleSearch({
                "q": query,
                "api_key": self.serpapi_key,
                "num": 5
            })
            results = search.get_dict()
            snippets = []
            if "answer_box" in results:
                box = results["answer_box"]
                if "answer" in box:
                    snippets.append(f"Direct answer: {box['answer']}")
                elif "snippet" in box:
                    snippets.append(f"Summary: {box['snippet']}")
            if "organic_results" in results:
                for r in results["organic_results"][:4]:
                    if "snippet" in r:
                        snippets.append(f"{r['title']}: {r['snippet']}")
            return "\n\n".join(snippets) if snippets else "No results found"
        except Exception as e:
            return f"Web search error: {e}"

    def calculate(self, expression):
        """Run a math calculation."""
        print(f"\n🔧 Calculating: {expression}")
        try:
            result = eval(expression)
            return f"**Result:** {expression} = **{result}**"
        except Exception as e:
            return f"Error: {e}"

    # ─────────────────────────────────────────
    # DECIDE WHICH TOOL TO USE
    # ─────────────────────────────────────────

    def decide_action(self, question):
        """Ask LLM to decide what action to take."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a routing assistant. Given a question decide what action to take.

Reply with EXACTLY one of these formats:
- SEARCH_DOCS: <search query>     → for questions about uploaded documents
- SEARCH_WEB: <search query>      → for live data, news, stock prices, current events
- CALCULATE: <math expression>    → for math calculations
- ANSWER: <direct answer>         → for simple questions you can answer directly

Examples:
"What was Apple's revenue?" → SEARCH_DOCS: Apple revenue 2023
"What is Apple's stock price today?" → SEARCH_WEB: Apple stock price today
"What is 25 * 4.5?" → CALCULATE: 25 * 4.5
"What is RAG?" → SEARCH_DOCS: RAG retrieval augmented generation"""
                },
                {
                    "role": "user",
                    "content": f"Conversation history:\n{json.dumps(self.history[-4:], indent=2)}\n\nQuestion: {question}"
                }
            ]
        )
        decision = response.choices[0].message.content.strip()
        print(f"\n🤔 Decision: {decision}")
        return decision

    # ─────────────────────────────────────────
    # MAIN ASK
    # ─────────────────────────────────────────

    def ask(self, question):
        self.history.append({"role": "user", "content": question})

        # Step 1: Decide what to do
        decision = self.decide_action(question)

        # Step 2: Execute the right tool
        if decision.startswith("CALCULATE:"):
            expression = decision.replace("CALCULATE:", "").strip()
            context = self.calculate(expression)
            answer = context

        elif decision.startswith("SEARCH_WEB:"):
            query = decision.replace("SEARCH_WEB:", "").strip()
            context = self.search_web(query)
            system_prompt = f"""You are an expert analyst assistant.
Answer the question using the web search results below.
Be clear, direct and helpful. Use **bold** for key facts and numbers.

WEB SEARCH RESULTS:
{context}"""
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.history
                ]
            )
            answer = response.choices[0].message.content

        elif decision.startswith("ANSWER:"):
            answer = decision.replace("ANSWER:", "").strip()

        else:
            # Default: SEARCH_DOCS
            if decision.startswith("SEARCH_DOCS:"):
                query = decision.replace("SEARCH_DOCS:", "").strip()
            else:
                query = question
            context = self.search_documents(query)
            system_prompt = f"""You are an expert analyst assistant — precise, clear, and helpful.

RESPONSE STYLE:
- Start with a direct answer immediately
- Use **bold** for key numbers and important facts
- Use bullet points for lists
- Use tables when comparing multiple values
- Keep responses focused and concise
- End complex answers with "**Summary:**" on a new line
- Never start with "Based on the context" or "According to the document"
- Never make up numbers or facts not in the context

CONTEXT FROM DOCUMENT:
{context}"""
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.history
                ]
            )
            answer = response.choices[0].message.content

        self.history.append({"role": "assistant", "content": answer})
        print(f"\n🤖 Agent: {answer}")
        return answer