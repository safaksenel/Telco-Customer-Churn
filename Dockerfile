FROM python:3.10-slim

WORKDIR /app

# Gerekli dosyaları kopyala
COPY requirements.txt .

# Bağımlılıkları yükle
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını ve veri setini kopyala
COPY . .

# Eğer model.joblib yoksa, container ayağa kalkarken önce eğitim yapsın
# (Opsiyonel olarak bu adımı dışarıda yapıp imaja sadece modeli de koyabilirsiniz
# Ancak "sadece veri seti var" dendiği için eğitimi burada tetikliyoruz)
RUN python train.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
