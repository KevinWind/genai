# Package Installation Guide for Jupyter Notebooks

## When running in `.venv` environment

### Method 1: Install in Notebook Cell (Quick fixes)
```python
# Most reliable method - uses the exact Python interpreter
import sys
!{sys.executable} -m pip install package_name

# Alternative (simpler but less reliable)
!pip install package_name

# Example: Installing transformers
import sys
!{sys.executable} -m pip install transformers

# Then import and use
import transformers
```

### Method 2: Terminal Installation (Project setup)
```bash
# Make sure you're in the right environment
source .venv/bin/activate

# Install packages
pip install package_name

# Update requirements
pip freeze > requirements-venv.txt
```

### Method 3: Requirements-based Installation
```python
# In notebook cell, install from requirements
import sys
!{sys.executable} -m pip install -r requirements-venv.txt
```

## Best Practices

1. **Always verify your environment**:
```python
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Check if in virtual environment
import os
venv = os.environ.get('VIRTUAL_ENV')
if venv:
    print(f"Virtual environment: {venv}")
else:
    print("Not in a virtual environment")
```

2. **For project dependencies, update requirements**:
```bash
pip freeze > requirements-venv.txt
```

3. **Check if package is already installed**:
```python
try:
    import package_name
    print("✅ Package already installed")
except ImportError:
    print("❌ Package not found, installing...")
    import sys
    !{sys.executable} -m pip install package_name
```

## Common Packages for AI/ML Projects
```python
# Hugging Face ecosystem
!{sys.executable} -m pip install transformers datasets accelerate

# PyTorch
!{sys.executable} -m pip install torch torchvision torchaudio

# Other common packages
!{sys.executable} -m pip install pandas numpy matplotlib seaborn scikit-learn
``` 