#!/bin/bash

echo "🚀 Initializing META-AI Production Environment..."

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️ .env not found. Creating from .env.example..."
    cp .env.example .env
    echo "🛠️ Please edit .env with your actual API keys."
else
    echo "✅ .env file detected."
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/vectordb data/sessions data/uploads

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo -e "\n✨ Setup Complete!"
echo "🏃 To start the application:"
echo "   1. Run the API:  uvicorn app.main:app --reload"
echo "   2. Run the UI:   streamlit run frontend/streamlit_app.py"
echo -e "\n🐳 Or use Docker:"
echo "   docker-compose up --build"
