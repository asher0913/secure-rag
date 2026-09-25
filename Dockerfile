FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
EXPOSE 8000
# Needs SECURE_RAG_TOKEN_SECRET at run time, e.g. docker run -e SECURE_RAG_TOKEN_SECRET=... -p 8000:8000 secure-rag
CMD ["uvicorn", "secure_rag.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

