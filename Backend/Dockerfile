FROM --platform=linux/amd64 python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout=30 --index-url https://mirrors.aliyun.com/pypi/simple/


# Copy FastAPI code
COPY . .

# Expose port 8000
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
