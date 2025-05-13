FROM python:3.10-slim

# Create a non-root user
RUN adduser --disabled-password --gecos '' sandbox

# Create workspace directory
WORKDIR /sandbox
RUN chown -R sandbox:sandbox /sandbox

# Install basic Python utilities
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn matplotlib

# Switch to non-root user
USER sandbox

# Copy executor code
COPY --chown=sandbox:sandbox executor.py /sandbox/

# Set the entry point
ENTRYPOINT ["python", "/sandbox/executor.py"] 