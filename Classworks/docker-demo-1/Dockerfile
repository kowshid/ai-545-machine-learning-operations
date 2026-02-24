# Dockerfile: minimal environment versioning example
# 1. Start from an official Python base image
FROM python:3.11-alpine
# 2. Set the working directory inside the container
WORKDIR /app
# 3. Copy dependency file and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 4. Copy the rest of the application code
COPY . .
# 5. Define the default command to run when the container starts
CMD ["python", "train.py"]