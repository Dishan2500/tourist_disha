FROM python:3.10-slim

WORKDIR /app

# Copy .env first so it's available to all subsequent steps
COPY .env .env

# Copy and install dependencies
COPY deployment/requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all repo files
COPY . /app

EXPOSE 7860

# Source .env and run Streamlit app
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["source /app/.env 2>/dev/null || true && \
     export HF_TOKEN=${HF_TOKEN:-''} && \
     streamlit run /app/master_folder/deployment/app.py \
       --server.port 7860 \
       --server.address 0.0.0.0 \
       --logger.level=info"]
