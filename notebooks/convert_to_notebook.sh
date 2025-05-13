#!/bin/bash
# Convert Python script to Jupyter notebook

# Check if jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "Jupyter is not installed. Installing..."
    pip install jupyter
fi

# Convert Python file to notebook
jupyter nbconvert --to notebook --execute notebooks/quickstart.py

echo "Notebook created: notebooks/quickstart.nbconvert.ipynb" 