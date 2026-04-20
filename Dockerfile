FROM python:3.11-slim

WORKDIR /app

# Bağımlılıklar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Veri ön işleme ve model eğitimi
RUN python 01_data_preprocessing.py && python 02_model_training.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
