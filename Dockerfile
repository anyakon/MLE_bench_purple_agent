FROM ghcr.io/anyakon/mlebench-env:latest

# Install uv
RUN pip install uv

# Copy project files
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies via uv
RUN uv sync --frozen

USER nonroot
ENTRYPOINT ["/bin/bash", "-lc", "cd /app && uv run src/server.py --host 0.0.0.0"]
EXPOSE 9009
