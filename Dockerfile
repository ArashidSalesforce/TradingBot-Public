# Base Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TA-Lib and other required tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    wget \
    make \
    libtool \
    automake \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr \
    && make && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Entry point to start the bot
CMD ["python", "bot.py"]



