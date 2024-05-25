# Sử dụng image Python làm nền tảng
FROM python:3.8-slim

# Cài đặt các gói phụ thuộc
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
COPY . /app

# Chạy ứng dụng Streamlit
CMD ["streamlit", "run", "stream_app.py"]
