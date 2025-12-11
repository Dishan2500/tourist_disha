# -------------------------
#  Base Python Image
# -------------------------
FROM python:3.10-slim

# -------------------------
#  Setup Working Directory
# -------------------------
WORKDIR /app

# -------------------------
#  Copy Project Files
# -------------------------
COPY . /app

# -------------------------
#  Install Dependencies
# -------------------------
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
#  Expose Streamlit Port
# -------------------------
EXPOSE 8501

# -------------------------
#  Run Streamlit App
# -------------------------
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
