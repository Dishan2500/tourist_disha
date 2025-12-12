FROM python:3.10

# Install dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . /app

# Expose the correct port
EXPOSE 7860

# Hugging Face expects to run: streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
CMD ["bash", "-c", "streamlit run app.py --server.address 0.0.0.0 --server.port $PORT"]
