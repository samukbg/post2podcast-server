FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install sqlite3 command-line tool
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Create directories for uploads and audio output
RUN mkdir -p uploads audio_output

# Expose the port the app runs on
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV UPLOAD_DIR=/app/uploads
ENV AUDIO_DIR=/app/audio_output

# Command to run the application
CMD ["python", "server.py"]
