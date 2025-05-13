#!/bin/bash
# Run all tests for the AZRL project

set -e  # Exit on error

echo "Running AZRL tests..."

# Discover and run all tests
python -m unittest discover -s tests

echo "All tests completed." 