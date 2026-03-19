FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

RUN mkdir -p quant_rag_agent/data quant_rag_agent/db quant_rag_agent/static

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
