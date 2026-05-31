#!/bin/bash
# Installation script for MLOps optimization

set -e

echo "🚀 Setting up Tourism Project MLOps Structure..."

# Create directories
echo "📁 Creating directory structure..."
mkdir -p src/{data,models,registry,hosting,utils}
mkdir -p tests
mkdir -p config
mkdir -p docs
mkdir -p .github/workflows

echo "✅ Directory structure created"
echo ""
echo "📝 Next steps:"
echo "1. Create all Python files in src/ directories"
echo "2. Create test files in tests/"
echo "3. Create workflow files in .github/workflows/"
echo "4. Update README.md"
echo "5. Run: git add . && git commit -m 'MLOps optimization'"
echo "6. Push: git push origin feat/mlops-optimization"
echo ""
echo "📚 Refer to IMPLEMENTATION.md for detailed file contents"
