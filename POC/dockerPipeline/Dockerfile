FROM --platform=linux/amd64 python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Install Supervisor
RUN apt-get update && apt-get install -y supervisor

# Copy Supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run Supervisor
CMD ["/usr/bin/supervisord"]
