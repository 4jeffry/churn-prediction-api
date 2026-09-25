FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file dan folder pendukung ke dalam container
COPY app.py .
COPY monitor.py .
COPY data/ data/
COPY models/ models/

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
