FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download NLTK punkt_tab
RUN python -c "import nltk; nltk.download('punkt_tab')"
COPY next_word_model.h5 .
COPY vocab.json .
COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]