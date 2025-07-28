# Adobe Hackathon Round 1B - Persona-Driven Document Intelligence Dockerfile
# Platform: linux/amd64, CPU-only, Offline execution
# Constraints: ≤60s processing, ≤1GB models, 3-5 document collections

FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (enhanced for 1B requirements)
RUN apt-get update && apt-get install -y \
    # Required for PyMuPDF
    libmupdf-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libopenjp2-7-dev \
    libjbig2dec0-dev \
    # Required for NLP and ML packages
    gcc \
    g++ \
    build-essential \
    # For spaCy and NLTK language models
    curl \
    wget \
    # For multiprocessing and system monitoring
    htop \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy 1B requirements first (for better Docker layer caching)
COPY requirements.txt /app/

# Install Python dependencies for Round 1B
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download and cache models for Round 1B (up to 1GB constraint)
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
import os; \
import nltk; \
os.makedirs('./models/tinybert/', exist_ok=True); \
os.makedirs('./models/cache/', exist_ok=True); \
os.makedirs('./models/nltk_data/', exist_ok=True); \
print('Downloading TinyBERT model for 1B...'); \
model_name = 'huawei-noah/TinyBERT_General_4L_312D'; \
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models/cache/'); \
model = AutoModel.from_pretrained(model_name, cache_dir='./models/cache/'); \
tokenizer.save_pretrained('./models/tinybert/'); \
model.save_pretrained('./models/tinybert/'); \
print('TinyBERT cached successfully!'); \
print('Downloading NLTK data...'); \
nltk.download('punkt', download_dir='./models/nltk_data/'); \
nltk.download('stopwords', download_dir='./models/nltk_data/'); \
nltk.download('wordnet', download_dir='./models/nltk_data/'); \
print('NLTK data cached successfully!');"

# Optional: Download spaCy model if needed (within 1GB constraint)
# Uncomment if you want to use spaCy for advanced NLP
# RUN python -m spacy download en_core_web_sm && \
#     mv /usr/local/lib/python3.10/site-packages/en_core_web_sm* ./models/

# Copy source code for Round 1B
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY main_1b.py /app/

# Set Python path and NLTK data path
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV NLTK_DATA="/app/models/nltk_data"

# Create input/output directories (Docker mount points)
RUN mkdir -p /app/input /app/output

# Set environment variables optimized for Round 1B
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=8
# Round 1B specific: Allow more CPU threads for multi-document processing
ENV NUMBA_NUM_THREADS=8
ENV MKL_NUM_THREADS=8

# Health check to verify Round 1B container is working
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "\
import sys; \
sys.path.insert(0, '/app/src'); \
try: \
    from persona_intelligence import PersonaManager, RelevanceScorer, MultiDocumentProcessor; \
    from pdf_intelligence.parser import PDFOutlineExtractor; \
    print('Container healthy: Round 1B Persona Intelligence loaded'); \
    exit(0); \
except Exception as e: \
    print(f'Container unhealthy: {e}'); \
    exit(1)"

# Default command for Round 1B (matches hackathon expected execution)
CMD ["python", "main_1b.py"]

# Metadata for Round 1B
LABEL org.opencontainers.image.title="Adobe Hackathon Round 1B - Persona-Driven Document Intelligence"
LABEL org.opencontainers.image.description="Multi-document intelligence with persona-based relevance scoring"
LABEL org.opencontainers.image.version="1.0"
LABEL hackathon.round="1B"
LABEL hackathon.team="Adopted"
LABEL hackathon.constraints="≤60s processing, ≤1GB models, 3-5 documents, CPU-only"
LABEL hackathon.features="persona-intelligence,multi-document,relevance-scoring,nlp-enhanced"