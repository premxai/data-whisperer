FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/vectorstore /app/outputs /app/outputs/logs
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 8501

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]
