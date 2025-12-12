FROM python:3.10-slim

WORKDIR /app

# Install minimal runtime dependencies for Streamlit UI
COPY deployment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy repo files (assumes app.py or Streamlit UI at repo root)
COPY . /app

# Load environment variables from .env if present
RUN if [ -f .env ]; then set -a && . ./.env && set +a; fi

EXPOSE 7860

CMD ["sh", "-c", "set -a && [ -f .env ] && . ./.env; set +a && streamlit run master_folder/deployment/app.py --server.port 7860 --server.address 0.0.0.0"]
