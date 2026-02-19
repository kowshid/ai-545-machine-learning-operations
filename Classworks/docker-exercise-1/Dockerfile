FROM python:3.11-alpine
WORKDIR /app
# Install system-level dependencies first (if needed)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
# Copy only dependency file
COPY requirements.txt .
#COPY . .
# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
# Copy source code
COPY . .
CMD ["python", "train.py"]