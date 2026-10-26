FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
