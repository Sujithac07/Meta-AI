#!/bin/bash

# AutoML Platform Pro - Complete Directory Structure Setup
# This script creates the enterprise-grade architecture

echo "🚀 Creating AutoML Platform Pro Directory Structure..."

# Create main directories
mkdir -p frontend/{src/{components,pages,services,styles},public}
mkdir -p backend/{api/routes,core,database,websocket,config}
mkdir -p ml_core/{auto_ml,features,training,evaluation,prediction,models}
mkdir -p deployment/{docker,kubernetes,cloud,scripts,ci_cd/{terraform,.github/workflows}}
mkdir -p monitoring/{dashboards,metrics,alerts,drift,logging}
mkdir -p explainability/{shap_integration,bias_detection,causal,interpretability}
mkdir -p tests/{unit,integration,performance}
mkdir -p docs
mkdir -p scripts
mkdir -p config
mkdir -p data/{raw,processed,models,cache}
mkdir -p logs

echo "✅ Directory structure created!"

# Create .gitkeep files for empty directories
find . -type d -empty -exec touch {}/.gitkeep \;

echo "✅ Git placeholders added!"

# Create main config files
cat > .env.example << 'EOF'
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/automl
REDIS_URL=redis://localhost:6379

# API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

# ML
ML_BACKEND=sklearn
WORKERS=4

# Deployment
ENVIRONMENT=development
LOG_LEVEL=INFO
EOF

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
env/

# Node
node_modules/
npm-debug.log
.next/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Environment
.env
.env.local
.env.*.local

# Logs
logs/
*.log

# Data
data/raw/
data/processed/
data/models/*.pkl
data/models/*.h5

# Testing
.pytest_cache/
htmlcov/

# System
.DS_Store
*.tmp
EOF

cat > README.md << 'EOF'
# 🚀 AutoML Platform Pro

Enterprise-grade Automated Machine Learning Platform with intelligent pipeline generation, production monitoring, and full explainability.

## 🎯 Features

- **Smart Pipeline Generator** - AI-powered architecture selection
- **Drag-Drop Pipeline Designer** - Visual ML engineering
- **Production Monitoring** - Real-time dashboards and alerts
- **Explainability** - SHAP, LIME, causal inference
- **Kubernetes Ready** - Enterprise deployment
- **Full API** - REST + WebSocket

## 🏗️ Architecture

```
AutoML-Platform-Pro/
├── frontend/          # React/Next.js dashboard
├── backend/           # FastAPI REST API
├── ml_core/           # ML algorithms & training
├── deployment/        # Docker, K8s, Cloud configs
├── monitoring/        # Prometheus, Grafana, alerts
└── explainability/    # SHAP, bias detection, causality
```

## 🚀 Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
python api/main.py

# Frontend
cd frontend
npm install
npm run dev
```

## 📚 Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 📝 License

Apache License 2.0
EOF

cat > Makefile << 'EOF'
.PHONY: install dev test build deploy clean

install:
	pip install -r requirements.txt
	npm install --prefix frontend

dev:
	@echo "Starting development environment..."
	docker-compose up

test:
	pytest tests/ -v

build:
	docker build -t automl-platform:latest .

deploy:
	./deployment/scripts/deploy.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make dev        - Start development environment"
	@echo "  make test       - Run tests"
	@echo "  make build      - Build Docker image"
	@echo "  make deploy     - Deploy to production"
	@echo "  make clean      - Clean up"
EOF

echo "✅ Configuration files created!"
echo ""
echo "📁 Directory structure:"
tree -L 2 2>/dev/null || find . -maxdepth 2 -type d | grep -v ".git"

echo ""
echo "🎉 Setup complete! Start building your enterprise AutoML platform!"
echo ""
echo "Next steps:"
echo "1. Review the structure"
echo "2. Create backend routes in backend/api/routes/"
echo "3. Create frontend components in frontend/src/components/"
echo "4. Implement ML core in ml_core/"
echo "5. Deploy using deployment configs"
