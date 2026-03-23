<div align="center">

<img src="https://wshaik-finsight-ai.hf.space/static/favicon.png" alt="FinSight AI Logo" width="100" />

# 🤖 FinSight AI

### Your AI-Powered Financial Research Assistant

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-wshaik--finsight--ai.hf.space-00D46A?style=for-the-badge)](https://wshaik-finsight-ai.hf.space/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*Ask questions about stocks, run backtests, compare companies, optimize portfolios, and analyze financial documents — all powered by AI.*

**[🌐 Try Live Demo](https://wshaik-finsight-ai.hf.space/) · [📝 Report Bug](https://github.com/WajidShaik10/finsight-ai/issues) · [✨ Request Feature](https://github.com/WajidShaik10/finsight-ai/issues)**

</div>

---

## 📸 Demo

> **Live at:** [https://wshaik-finsight-ai.hf.space/](https://wshaik-finsight-ai.hf.space/)

The app features a sleek dark-themed chat interface with 4 core capabilities accessible from the home screen:

| 📈 Stock Price | ⚖️ Compare | 🎯 Optimize | 📄 Analyze Doc |
|---|---|---|---|
| Get live prices & charts | Side-by-side company comparison | Portfolio weight optimization | Upload & query 10-K filings |

---

## ✨ Features

- **💬 Conversational AI** — Natural language interface powered by Groq's ultra-fast LLM inference
- **📊 Live Market Data** — Real-time stock prices, historical data, and charts via yFinance
- **🔍 RAG (Retrieval-Augmented Generation)** — Upload financial documents (10-K, PDFs) and ask questions grounded in the actual content
- **📉 Backtesting Engine** — Test trading strategies against historical market data
- **⚖️ Company Comparison** — Side-by-side fundamental analysis of multiple stocks
- **🎯 Portfolio Optimization** — Modern portfolio theory-based weight optimization using SciPy
- **💾 Persistent Chat History** — All conversations saved and resumable across sessions
- **🐳 Docker & Cloud Ready** — Deployed on Hugging Face Spaces; fully containerized

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, Python 3.10+ |
| **LLM Inference** | Groq (Llama 3) |
| **Vector Database** | ChromaDB |
| **Embeddings** | Sentence Transformers |
| **Market Data** | yFinance |
| **Document Parsing** | pdfplumber |
| **Portfolio Math** | NumPy, SciPy, Pandas |
| **Web Search** | SerpAPI |
| **Containerization** | Docker |
| **Deployment** | Hugging Face Spaces |

---

## 🏗️ Architecture

```
User (Browser)
     │
     ▼
Frontend (HTML/CSS/JS)
     │
     ▼
FastAPI Backend (api.py)
     ├── /chats/create     → Create new chat session
     ├── /chats/message    → Send message to AI agent
     ├── /chats/{id}/upload → Upload document for RAG
     └── /chats            → List all chat sessions
          │
          ├── QuantAgent (quant_rag_agent/modules/agent.py)
          │    ├── Groq LLM (llama-3 inference)
          │    ├── ChromaDB (vector retrieval)
          │    ├── yFinance (live market data)
          │    └── SerpAPI (web search)
          │
          └── DocumentIngester (quant_rag_agent/modules/ingester.py)
               ├── PDF parsing (pdfplumber)
               ├── HTML parsing
               └── Sentence Transformer embeddings → ChromaDB
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com/) (free tier available)
- [SerpAPI Key](https://serpapi.com/) (optional, for web search)

### 1. Clone the Repository

```bash
git clone https://github.com/WajidShaik10/finsight-ai.git
cd finsight-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here   # optional
```

### 4. Run the Application

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at [http://localhost:8000](http://localhost:8000)

### 5. (Optional) Run with Docker

```bash
docker build -t finsight-ai .
docker run -p 8000:8000 --env-file .env finsight-ai
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the chat UI |
| `GET` | `/documents` | List uploaded documents |
| `POST` | `/upload` | Upload a document |
| `POST` | `/chats/create` | Create a new chat session |
| `GET` | `/chats` | List all chat sessions |
| `POST` | `/chats/message` | Send a message to a chat |
| `POST` | `/chats/{id}/upload` | Upload document to a specific chat |
| `DELETE` | `/chats/{id}` | Delete a chat session |

---

## 💡 Example Queries

```
"What is Apple's current stock price?"
"Compare Tesla and Rivian revenue growth over 3 years"
"Optimize a portfolio of AAPL, MSFT, GOOGL, and AMZN"
"What does this 10-K say about risk factors?"
"Backtest a moving average crossover strategy on NVDA"
```

---

## 📁 Project Structure

```
finsight-ai/
├── api.py                    # Main FastAPI application & REST endpoints
├── main.py                   # CLI entry point for local testing
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container configuration
├── Procfile                  # Process configuration for deployment
├── render.yaml               # Render.com deployment config
├── quant_rag_agent/
│   ├── modules/
│   │   ├── agent.py          # Core AI agent (Groq LLM + tools)
│   │   └── ingester.py       # Document ingestion & embedding pipeline
│   ├── static/               # Frontend HTML/CSS/JS
│   └── data/                 # Uploaded documents & chat history
└── frontend/                 # Additional frontend assets
```

---

## 🔮 Roadmap

- [ ] 📊 Interactive stock charts in chat
- [ ] 🔔 Price alerts and notifications
- [ ] 📰 Real-time financial news integration
- [ ] 🌐 Multi-language support
- [ ] 📱 Mobile-optimized UI
- [ ] 🔐 User authentication & personal portfolios
- [ ] 📤 Export reports as PDF

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**Made with ❤️ by [Wajid Shaik](https://github.com/WajidShaik10)**

⭐ Star this repo if you find it useful!

[![GitHub stars](https://img.shields.io/github/stars/WajidShaik10/finsight-ai?style=social)](https://github.com/WajidShaik10/finsight-ai/stargazers)

</div>
