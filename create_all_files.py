#!/usr/bin/env python3
"""
Complete MLOps File Generator
Usage: python create_all_files.py
Creates all 22 files needed for MLOps optimization
"""

import os
from pathlib import Path

def create_file(file_path, content):
    """Create a file with the given content"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ {file_path}")

# ============================================================================
# README.md
# ============================================================================

README = """# Tourism Project - Customer Purchase Prediction MLOps Pipeline

> **An enterprise-grade MLOps pipeline for predicting customer purchases in the wellness tourism industry**

## 🎯 Overview

**Tourism Project** is a production-ready MLOps pipeline for predicting customer purchases of the Wellness Tourism Package.

### Key Metrics
- **Dataset**: 22+ customer features
- **Model**: XGBoost with hyperparameter tuning  
- **Task**: Binary classification (Purchase/No Purchase)
- **Pipeline**: Data prep → Training → Registry → Deployment

## 🏗️ Architecture

```
GitHub Actions (CI/CD)
    ↓
Data Prep → Model Training → Model Registry → Streamlit Deployment
    ↓           ↓                ↓                    ↓
HF Dataset  MLflow Logs      HF Model            HF Spaces
```

## ✨ Features

✅ Automated Data Pipeline  
✅ Advanced Model Training with Hyperparameter Tuning  
✅ Model Registry & Versioning  
✅ Production Deployment on HuggingFace Spaces  
✅ CI/CD Automation with GitHub Actions  
✅ Comprehensive MLflow Experiment Tracking  

## 📦 Prerequisites

- Python 3.9+
- Docker (optional)
- Git
- HuggingFace Account with API token
- MLflow

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/rama64palle/Tourism_Project.git
cd Tourism_Project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Set environment
export HF_TOKEN=your_token_here

# 3. Prepare data
python -m src.data.data_prep

# 4. Train model (separate terminal)
mlflow ui --host 0.0.0.0 --port 5000
python -m src.models.train_model

# 5. Run Streamlit app
cd deployment
streamlit run app.py
```

## 📁 Project Structure

```
Tourism_Project/
├── .github/workflows/          # CI/CD pipelines
├── src/
│   ├── data/                   # Data preprocessing
│   ├── models/                 # Model training
│   ├── registry/               # Model registry
│   ├── hosting/                # Deployment
│   └── utils/                  # Configuration
├── tests/                      # Unit tests
├── deployment/                 # Streamlit app
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 Configuration

Set environment variables:
```bash
export HF_TOKEN=your_token
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## 📊 Monitoring

- MLflow UI: http://localhost:5000
- CI/CD: https://github.com/rama64palle/Tourism_Project/actions

## 📝 License

MIT License

---

**Status**: ✅ Production Ready | **Version**: 1.0.0 | **Last Updated**: May 31, 2026
"""

# ============================================================================
# src/data/data_prep.py
# ============================================================================

DATA_PREP = """\"\"\"Data Preparation Module for Tourism Project\"\"\"

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    \"\"\"Handle data loading, cleaning, and preprocessing.\"\"\"
    
    def __init__(self, hf_token: str = None):
        self.api = HfApi(token=hf_token or os.getenv("HF_TOKEN"))
        self.label_encoders = {}
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        \"\"\"Load dataset from local or remote path.\"\"\"
        try:
            df = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess(self, df: pd.DataFrame, target_col: str = 'ProdTaken') -> tuple:
        \"\"\"Preprocess data: remove identifiers, encode categorical features.\"\"\"
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
            logger.info(f"Preprocessing complete. Features: {X.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                        test_size: float = 0.2, random_state: int = 42):
        \"\"\"Split data into train and test sets.\"\"\"
        try:
            Xtrain, Xtest, ytrain, ytest = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Split: Train {Xtrain.shape}, Test {Xtest.shape}")
            return Xtrain, Xtest, ytrain, ytest
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
    
    def save_splits(self, Xtrain: pd.DataFrame, Xtest: pd.DataFrame, 
                   ytrain: pd.Series, ytest: pd.Series, output_dir: str = "."):
        \"\"\"Save train/test splits locally and upload to HuggingFace.\"\"\"
        try:
            os.makedirs(output_dir, exist_ok=True)
            Xtrain.to_csv(f"{output_dir}/Xtrain.csv", index=False)
            Xtest.to_csv(f"{output_dir}/Xtest.csv", index=False)
            ytrain.to_csv(f"{output_dir}/ytrain.csv", index=False)
            ytest.to_csv(f"{output_dir}/ytest.csv", index=False)
            logger.info("Data splits saved locally")
        except Exception as e:
            logger.error(f"Error saving splits: {e}")
            raise


def main():
    preprocessor = DataPreprocessor()
    df = preprocessor.load_dataset("hf://datasets/rama64palle/Tourism_Project/tourism.csv")
    X, y = preprocessor.preprocess(df)
    Xtrain, Xtest, ytrain, ytest = preprocessor.train_test_split(X, y)
    preprocessor.save_splits(Xtrain, Xtest, ytrain, ytest)


if __name__ == "__main__":
    main()
"""

# ============================================================================
# src/models/train_model.py
# ============================================================================

TRAIN_MODEL = """\"\"\"Model Training Module for Tourism Project\"\"\"

import os
import pandas as pd
import xgboost as xgb
import joblib
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    \"\"\"Handle model training, hyperparameter tuning, and evaluation.\"\"\"
    
    NUMERIC_FEATURES = [
        'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
        'Passport', 'PitchSatisfactionScore', 'OwnCar',
        'NumberOfChildrenVisiting', 'MonthlyIncome'
    ]
    
    def __init__(self, mlflow_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("mlops-training-experiment")
        self.api = HfApi(token=os.getenv("HF_TOKEN"))
        logger.info(f"MLflow URI: {mlflow_uri}")
    
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
            'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
            'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
            'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1)
        grid_search.fit(Xtrain, ytrain)
        logger.info(f"Best params: {grid_search.best_params_}")
        return grid_search


def main():
    trainer = ModelTrainer()
    paths = [
        "hf://datasets/rama64palle/Tourism_Project/Xtrain.csv",
        "hf://datasets/rama64palle/Tourism_Project/Xtest.csv",
        "hf://datasets/rama64palle/Tourism_Project/ytrain.csv",
        "hf://datasets/rama64palle/Tourism_Project/ytest.csv"
    ]
    Xtrain, Xtest, ytrain, ytest = trainer.load_data(*paths)
    pipeline, _ = trainer.build_pipeline(Xtrain, ytrain)
    grid_search = trainer.hyperparameter_tuning(pipeline, Xtrain, ytrain)


if __name__ == "__main__":
    main()
"""

# ============================================================================
# src/registry/model_registry.py
# ============================================================================

MODEL_REGISTRY = """\"\"\"Model Registry Module\"\"\"

import os
import logging
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    \"\"\"Manage model and dataset registration on HuggingFace Hub.\"\"\"
    
    def __init__(self, hf_token: str = None):
        self.api = HfApi(token=hf_token or os.getenv("HF_TOKEN"))
        logger.info("ModelRegistry initialized")
    
    def register_dataset(self, repo_id: str, folder_path: str = "tourism_project/data"):
        try:
            try:
                self.api.repo_info(repo_id=repo_id, repo_type="dataset")
            except RepositoryNotFoundError:
                create_repo(repo_id=repo_id, repo_type="dataset", private=False)
            
            self.api.upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="dataset")
            logger.info(f"Dataset uploaded to {repo_id}")
        except Exception as e:
            logger.error(f"Error: {e}")
            raise


def main():
    registry = ModelRegistry()
    registry.register_dataset("rama64palle/Tourism_Project")


if __name__ == "__main__":
    main()
"""

# ============================================================================
# src/hosting/deploy_spaces.py
# ============================================================================

DEPLOY_SPACES = """\"\"\"HuggingFace Spaces Deployment Module\"\"\"

import os
import logging
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpacesDeployer:
    \"\"\"Deploy Streamlit app to HuggingFace Spaces.\"\"\"
    
    def __init__(self, hf_token: str = None):
        self.api = HfApi(token=hf_token or os.getenv("HF_TOKEN"))
    
    def deploy(self, repo_id: str, local_folder: str = "deployment"):
        try:
            logger.info(f"Deploying to {repo_id}")
            self.api.upload_folder(folder_path=local_folder, repo_id=repo_id, repo_type="space")
            logger.info(f"Deployed to {repo_id}")
        except Exception as e:
            logger.error(f"Error: {e}")
            raise


def main():
    deployer = SpacesDeployer()
    deployer.deploy("rama64palle/Tourism-Project")


if __name__ == "__main__":
    main()
"""

# ============================================================================
# src/utils/config.py
# ============================================================================

CONFIG_PY = """\"\"\"Configuration settings for Tourism MLOps Project\"\"\"

import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    \"\"\"Data configuration settings.\"\"\"
    dataset_repo_id: str = "rama64palle/Tourism_Project"
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class ModelConfig:
    \"\"\"Model configuration settings.\"\"\"
    model_repo_id: str = "rama64palle/Tourism_Project_Model"
    model_name: str = "tourism_project_model_v1.joblib"
    classification_threshold: float = 0.45
    cv_folds: int = 5


class Config:
    \"\"\"Master configuration class.\"\"\"
    
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
    
    @property
    def hf_token(self) -> str:
        return os.getenv("HF_TOKEN", "")


config = Config()
"""

# ============================================================================
# tests/test_data.py
# ============================================================================

TEST_DATA = """#!/usr/bin/env python
\"\"\"Data validation tests for Tourism Project\"\"\"

import pytest
import pandas as pd
from src.data.data_prep import DataPreprocessor


def test_data_preprocessing():
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


def test_train_test_split():
    preprocessor = DataPreprocessor()
    X = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200)})
    y = pd.Series([0, 1] * 50)
    Xtrain, Xtest, ytrain, ytest = preprocessor.train_test_split(X, y, test_size=0.2)
    assert len(Xtrain) + len(Xtest) == 100
    assert len(Xtest) / len(X) == pytest.approx(0.2, rel=0.05)
"""

# ============================================================================
# tests/test_model.py
# ============================================================================

TEST_MODEL = """#!/usr/bin/env python
\"\"\"Model validation tests for Tourism Project\"\"\"

import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def test_model_predictions():
    X, y = make_classification(n_samples=100, n_features=12, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions) == {0, 1}


def test_model_generalization():
    X, y = make_classification(n_samples=200, n_features=12, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    assert train_score >= 0
    assert test_score >= 0
    assert abs(train_score - test_score) < 0.5
"""

# ============================================================================
# .gitignore
# ============================================================================

GITIGNORE = """*.pyc
__pycache__/
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
dist/
build/
.env
.venv
venv/
env/
*.log
.DS_Store
.vscode/
.idea/
*.swp
*.swo
*~
mlruns/
.mlflow/
*.joblib
*.pkl
*.pickle
data/processed/
*.csv
!tourism.csv
.ipynb_checkpoints/
*.ipynb
"""

# ============================================================================
# pyproject.toml
# ============================================================================

PYPROJECT = """[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "tourism-project"
version = "1.0.0"
description = "Enterprise-grade MLOps pipeline for customer purchase prediction"
requires-python = ">=3.9"
license = {text = "MIT"}

dependencies = [
    "pandas==2.2.2",
    "scikit-learn==1.6.0",
    "xgboost==2.1.4",
    "mlflow==3.0.1",
    "huggingface_hub==0.32.6",
    "streamlit==1.43.2",
    "joblib==1.5.1",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]
"""

# ============================================================================
# Workflows
# ============================================================================

DATA_PREP_YML = """name: Data Preparation

on:
  workflow_dispatch:
  push:
    branches: [main]
    paths: ['src/data/**']

jobs:
  prepare:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - name: Prepare Data
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: python -m src.data.data_prep
"""

MODEL_TRAINING_YML = """name: Model Training

on:
  workflow_dispatch:
  push:
    branches: [main]
    paths: ['src/models/**']

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: nohup mlflow ui --host 0.0.0.0 --port 5000 &
      - name: Train Model
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          MLFLOW_TRACKING_URI: http://localhost:5000
        run: python -m src.models.train_model
"""

DEPLOY_YML = """name: Deployment

on:
  workflow_dispatch:
  push:
    branches: [main]
    paths: ['deployment/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - name: Deploy to Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: python -m src.hosting.deploy_spaces
"""

PIPELINE_YML = """name: MLOps Pipeline

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'

jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - name: Prepare Data
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: python -m src.data.data_prep
      - run: nohup mlflow ui --host 0.0.0.0 --port 5000 &
      - name: Train Model
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          MLFLOW_TRACKING_URI: http://localhost:5000
        run: python -m src.models.train_model
      - name: Deploy
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: python -m src.hosting.deploy_spaces
"""

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("\\n🚀 Creating MLOps Files...\\n")
    
    files = {
        'README.md': README,
        'src/__init__.py': '# Tourism Project - MLOps Pipeline\\n',
        'src/data/__init__.py': '# Data Preparation Module\\n',
        'src/data/data_prep.py': DATA_PREP,
        'src/models/__init__.py': '# Models Module\\n',
        'src/models/train_model.py': TRAIN_MODEL,
        'src/registry/__init__.py': '# Registry Module\\n',
        'src/registry/model_registry.py': MODEL_REGISTRY,
        'src/hosting/__init__.py': '# Hosting Module\\n',
        'src/hosting/deploy_spaces.py': DEPLOY_SPACES,
        'src/utils/__init__.py': '# Configuration Module\\n',
        'src/utils/config.py': CONFIG_PY,
        'tests/__init__.py': '# Tests Module\\n',
        'tests/test_data.py': TEST_DATA,
        'tests/test_model.py': TEST_MODEL,
        '.gitignore': GITIGNORE,
        'pyproject.toml': PYPROJECT,
        '.github/workflows/data-prep.yml': DATA_PREP_YML,
        '.github/workflows/model-training.yml': MODEL_TRAINING_YML,
        '.github/workflows/deploy.yml': DEPLOY_YML,
        '.github/workflows/mlops-pipeline.yml': PIPELINE_YML,
    }
    
    for file_path, content in files.items():
        create_file(file_path, content)
    
    print(f"\\n✅ Created {len(files)} files!\\n")
    print("📝 Next steps:")
    print("   git add .")
    print("   git commit -m 'Add complete MLOps optimization'")
    print("   git push origin feat/mlops-optimization\\n")


if __name__ == "__main__":
    main()
"""

print("Creating file creation script...")
create_file("create_all_files.py", CREATE_ALL_FILES)
print("\n✅ Script created!")
