FROM python:3.10-slim

WORKDIR /app

# Install minimal runtime dependencies for Streamlit UI
COPY deployment/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy repo files (assumes app.py or Streamlit UI at repo root)
COPY . /app

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
