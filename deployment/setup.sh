#!/bin/bash
# Install Google Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt-get update && apt-get install -y ./google-chrome-stable_current_amd64.deb

# Install ChromeDriver
wget https://storage.googleapis.com/chrome-for-testing-public/121.0.6167.0/linux64/chromedriver-linux64.zip
unzip chromedriver-linux64.zip
mv chromedriver /usr/bin/chromedriver
chmod +x /usr/bin/chromedriver

# Install required libraries
apt-get install -y libglib2.0-0 libnss3 libx11-6 libatk1.0-0 libcups2 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libgtk-3-0 libasound2