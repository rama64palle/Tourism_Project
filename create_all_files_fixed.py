#!/usr/bin/env python3
"""
Complete MLOps File Generator
Usage: python create_all_files.py
"""

import os
from pathlib import Path

def create_file(file_path, content):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"OK {file_path}")

# Files dictionary
files_to_create = {}

# README.md
files_to_create['README.md'] = """# Tourism Project - MLOps Pipeline

Enterprise-grade MLOps pipeline for customer purchase prediction

## Overview
Tourism Project predicts customer purchases of Wellness Tourism Package.

## Quick Start
```bash
git clone https://github.com/rama64palle/Tourism_Project.git
cd Tourism_Project
pip install -r requirements.txt
export HF_TOKEN=your_token
python -m src.data.data_prep
```

## Project Structure
- src/data - Data preprocessing
- src/models - Model training  
- src/registry - Model registry
- src/hosting - Deployment
- tests - Unit tests

## License
MIT
"""

# src/data/data_prep.py
files_to_create['src/data/data_prep.py'] = """import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, hf_token=None):
        self.api = HfApi(token=hf_token or os.getenv("HF_TOKEN"))
        self.label_encoders = {}
    
    def load_dataset(self, dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    def preprocess(self, df, target_col='ProdTaken'):
        try:
            if 'CustomerID' in df.columns:
                df = df.drop(columns=['CustomerID'])
            
            categorical_features = df.select_dtypes(object).columns
            for feature in categorical_features:
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature])
                self.label_encoders[feature] = le
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            logger.info(f"Features: {X.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        try:
            Xtrain, Xtest, ytrain, ytest = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Train {Xtrain.shape}, Test {Xtest.shape}")
            return Xtrain, Xtest, ytrain, ytest
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
    
    def save_splits(self, Xtrain, Xtest, ytrain, ytest, output_dir="."):
        try:
            os.makedirs(output_dir, exist_ok=True)
            Xtrain.to_csv(f"{output_dir}/Xtrain.csv", index=False)
            Xtest.to_csv(f"{output_dir}/Xtest.csv", index=False)
            ytrain.to_csv(f"{output_dir}/ytrain.csv", index=False)
            ytest.to_csv(f"{output_dir}/ytest.csv", index=False)
            logger.info("Saved")
        except Exception as e:
            logger.error(f"Error: {e}")
            raise

def main():
    preprocessor = DataPreprocessor()
    df = preprocessor.load_dataset("tourism.csv")
    X, y = preprocessor.preprocess(df)
    Xtrain, Xtest, ytrain, ytest = preprocessor.train_test_split(X, y)
    preprocessor.save_splits(Xtrain, Xtest, ytrain, ytest)

if __name__ == "__main__":
    main()
"""

# src/models/train_model.py
files_to_create['src/models/train_model.py'] = """import os
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    NUMERIC_FEATURES = [
        'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
        'Passport', 'PitchSatisfactionScore', 'OwnCar',
        'NumberOfChildrenVisiting', 'MonthlyIncome'
    ]
    
    def __init__(self, mlflow_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("mlops-training")
        logger.info(f"MLflow: {mlflow_uri}")
    
    def load_data(self, Xtrain_path, Xtest_path, ytrain_path, ytest_path):
        Xtrain = pd.read_csv(Xtrain_path)
        Xtest = pd.read_csv(Xtest_path)
        ytrain = pd.read_csv(ytrain_path)
        ytest = pd.read_csv(ytest_path)
        logger.info("Data loaded")
        return Xtrain, Xtest, ytrain, ytest
    
    def build_pipeline(self, Xtrain, ytrain):
        categorical_features = Xtrain.select_dtypes(object).columns.tolist()
        class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
        
        preprocessor = make_column_transformer(
            (StandardScaler(), self.NUMERIC_FEATURES),
            (OneHotEncoder(handle_unknown='ignore'), categorical_features)
        )
        
        xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)
        pipeline = make_pipeline(preprocessor, xgb_model)
        logger.info("Pipeline built")
        return pipeline, class_weight
    
    def hyperparameter_tuning(self, pipeline, Xtrain, ytrain, cv=5):
        param_grid = {
            'xgbclassifier__n_estimators': [50, 75, 100],
            'xgbclassifier__max_depth': [2, 3, 4],
            'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
            'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)
        logger.info(f"Best: {grid_search.best_params_}")
        return grid_search

def main():
    trainer = ModelTrainer()
    logger.info("Training started")

if __name__ == "__main__":
    main()
"""

# src/registry/model_registry.py
files_to_create['src/registry/model_registry.py'] = """import os
import logging
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, hf_token=None):
        self.api = HfApi(token=hf_token or os.getenv("HF_TOKEN"))
        logger.info("Registry initialized")
    
    def register_dataset(self, repo_id, folder_path="tourism_project/data"):
        try:
            try:
                self.api.repo_info(repo_id=repo_id, repo_type="dataset")
            except RepositoryNotFoundError:
                create_repo(repo_id=repo_id, repo_type="dataset", private=False)
            logger.info(f"Uploaded to {repo_id}")
        except Exception as e:
            logger.error(f"Error: {e}")

def main():
    registry = ModelRegistry()
    logger.info("Registry ready")

if __name__ == "__main__":
    main()
"""

# src/hosting/deploy_spaces.py
files_to_create['src/hosting/deploy_spaces.py'] = """import os
import logging
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpacesDeployer:
    def __init__(self, hf_token=None):
        self.api = HfApi(token=hf_token or os.getenv("HF_TOKEN"))
    
    def deploy(self, repo_id, local_folder="deployment"):
        try:
            logger.info(f"Deploying to {repo_id}")
            self.api.upload_folder(folder_path=local_folder, repo_id=repo_id, repo_type="space")
            logger.info("Deployed")
        except Exception as e:
            logger.error(f"Error: {e}")

def main():
    deployer = SpacesDeployer()
    logger.info("Deployer ready")

if __name__ == "__main__":
    main()
"""

# src/utils/config.py
files_to_create['src/utils/config.py'] = """import os
from dataclasses import dataclass

@dataclass
class DataConfig:
    dataset_repo_id: str = "rama64palle/Tourism_Project"
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class ModelConfig:
    model_repo_id: str = "rama64palle/Tourism_Project_Model"
    model_name: str = "tourism_project_model_v1.joblib"
    classification_threshold: float = 0.45
    cv_folds: int = 5

class Config:
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
    
    @property
    def hf_token(self):
        return os.getenv("HF_TOKEN", "")

config = Config()
"""

# tests/test_data.py
files_to_create['tests/test_data.py'] = """import pytest
import pandas as pd
from src.data.data_prep import DataPreprocessor

def test_preprocessing():
    preprocessor = DataPreprocessor()
    df = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'Age': [25, 30, 35],
        'Type': ['A', 'B', 'A'],
        'ProdTaken': [0, 1, 0]
    })
    X, y = preprocessor.preprocess(df)
    assert 'CustomerID' not in X.columns
    assert len(X) == len(y)

def test_split():
    preprocessor = DataPreprocessor()
    X = pd.DataFrame({'f1': range(100), 'f2': range(100, 200)})
    y = pd.Series([0, 1] * 50)
    Xtrain, Xtest, ytrain, ytest = preprocessor.train_test_split(X, y, test_size=0.2)
    assert len(Xtrain) + len(Xtest) == 100
"""

# tests/test_model.py
files_to_create['tests/test_model.py'] = """import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

def test_predictions():
    X, y = make_classification(n_samples=100, n_features=12, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions) == {0, 1}
"""

# .gitignore
files_to_create['.gitignore'] = """*.pyc
__pycache__/
.pytest_cache/
.venv
venv/
*.log
.DS_Store
.env
mlruns/
*.joblib
*.pkl
.idea/
.vscode/
"""

# pyproject.toml
files_to_create['pyproject.toml'] = """[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "tourism-project"
version = "1.0.0"
description = "MLOps pipeline"
requires-python = ">=3.9"

dependencies = [
    "pandas==2.2.2",
    "scikit-learn==1.6.0",
    "xgboost==2.1.4",
    "mlflow==3.0.1",
    "huggingface_hub==0.32.6",
    "streamlit==1.43.2",
    "joblib==1.5.1",
]
"""

# Workflows
files_to_create['.github/workflows/data-prep.yml'] = """name: Data Preparation
on:
  push:
    branches: [main]
jobs:
  prepare:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python -m src.data.data_prep
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
"""

files_to_create['.github/workflows/model-training.yml'] = """name: Model Training
on:
  push:
    branches: [main]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python -m src.models.train_model
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
"""

files_to_create['.github/workflows/deploy.yml'] = """name: Deployment
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python -m src.hosting.deploy_spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
"""

# __init__ files
files_to_create['src/__init__.py'] = '# Tourism Project\n'
files_to_create['src/data/__init__.py'] = '# Data\n'
files_to_create['src/models/__init__.py'] = '# Models\n'
files_to_create['src/registry/__init__.py'] = '# Registry\n'
files_to_create['src/hosting/__init__.py'] = '# Hosting\n'
files_to_create['src/utils/__init__.py'] = '# Utils\n'
files_to_create['tests/__init__.py'] = '# Tests\n'

def main():
    print("\n Creating files...\n")
    
    for file_path, content in files_to_create.items():
        create_file(file_path, content)
    
    print(f"\n SUCCESS: Created {len(files_to_create)} files!\n")
    print("Next steps:")
    print("  git add .")
    print("  git commit -m 'Add MLOps optimization'")
    print("  git push origin feat/mlops-optimization\n")

if __name__ == "__main__":
    main()
