#!/usr/bin/env python3
import os
from pathlib import Path

def create_file(file_path, content):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"OK {file_path}")

print("Creating files...")

# README.md
create_file('README.md', """# Tourism Project MLOps Pipeline
Enterprise-grade MLOps pipeline for predicting customer purchases.
""")

# src/__init__.py files
create_file('src/__init__.py', '# Tourism Project')
create_file('src/data/__init__.py', '# Data module')
create_file('src/models/__init__.py', '# Models module')
create_file('src/registry/__init__.py', '# Registry module')
create_file('src/hosting/__init__.py', '# Hosting module')
create_file('src/utils/__init__.py', '# Utils module')
create_file('tests/__init__.py', '# Tests')

# src/data/data_prep.py
create_file('src/data/data_prep.py', """import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def load_dataset(self, path):
        return pd.read_csv(path)
    
    def preprocess(self, df):
        return df, df
""")

# src/models/train_model.py
create_file('src/models/train_model.py', """import logging
logger = logging.getLogger(__name__)

class ModelTrainer:
    def train(self):
        logger.info("Training model")
""")

# src/registry/model_registry.py
create_file('src/registry/model_registry.py', """import logging
logger = logging.getLogger(__name__)

class ModelRegistry:
    def register(self):
        logger.info("Registering model")
""")

# src/hosting/deploy_spaces.py
create_file('src/hosting/deploy_spaces.py', """import logging
logger = logging.getLogger(__name__)

class SpacesDeployer:
    def deploy(self):
        logger.info("Deploying")
""")

# src/utils/config.py
create_file('src/utils/config.py', """import os

class Config:
    hf_token = os.getenv('HF_TOKEN', '')
""")

# tests/test_data.py
create_file('tests/test_data.py', """def test_example():
    assert True
""")

# tests/test_model.py
create_file('tests/test_model.py', """def test_model():
    assert True
""")

# .gitignore
create_file('.gitignore', """*.pyc
__pycache__/
.venv
venv/
*.log
.env
""")

# pyproject.toml
create_file('pyproject.toml', """[project]
name = "tourism-project"
version = "1.0.0"
""")

# Workflows
create_file('.github/workflows/data-prep.yml', """name: Data Prep
on: [push]
jobs:
  prepare:
    runs-on: ubuntu-latest
""")

create_file('.github/workflows/train.yml', """name: Training
on: [push]
jobs:
  train:
    runs-on: ubuntu-latest
""")

create_file('.github/workflows/deploy.yml', """name: Deploy
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
""")

print("Done! All files created.")
print("Next: git add . && git commit -m 'Add files' && git push")
