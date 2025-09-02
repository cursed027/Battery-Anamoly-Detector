# Use slim Python base
FROM python:3.10-slim

WORKDIR /app

# Install system deps only if really needed (kept minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to leverage Docker cache)
COPY req.txt /app/req.txt

# Use pip cache for faster rebuilds
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r req.txt

# Copy rest of the app
COPY . /app

EXPOSE 8000 8501

# Run both FastAPI and Streamlit in same container
CMD ["bash", "-c", "\
    uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000 & \
    streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 \
"]
