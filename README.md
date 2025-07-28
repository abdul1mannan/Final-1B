# 🧠 Adobe India Hackathon 2025 - Challenge 1B: Dynamic Persona-Driven Document Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-1.26.3-green.svg)](https://pymupdf.readthedocs.io/)
[![TinyBERT](https://img.shields.io/badge/TinyBERT-4L%2F312D-orange.svg)](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)

A revolutionary dynamic document intelligence system that adapts to any persona without hardcoded templates, processing 3-15 PDF collections within 60 seconds with intelligent relevance scoring and pattern recognition.

## 📋 Submission Overview

- ✅ **Dynamic Persona System** - Handles any persona without hardcoding  
- ✅ **Performance Constraints**: ≤60s processing, ≤1GB models, ≤8GB RAM, offline operation
- ✅ **Intelligent Pattern Recognition** with adaptive content analysis
- ✅ **Sample Format Compliance** - Exact output matching Adobe specifications

## 🏗️ Our Revolutionary Approach

### Dynamic Intelligence Pipeline

```
Any Persona Input → Pattern Recognition → Multi-Document Processing → Adaptive Scoring → Structured Output
                           ↓
        ┌─────────────────────────────────────────────────────────────────┐
        │ 1. Dynamic Persona Recognition (Zero hardcoding, pattern-based) │
        │ 2. Intelligent Job Analysis (Automatic requirement extraction)   │ 
        │ 3. Adaptive Relevance Scoring (Context-aware ranking)           │
        │ 4. Format-Compliant Output (Exact sample structure matching)    │
        └─────────────────────────────────────────────────────────────────┘
```

## ✨ Key Innovations

### Dynamic Persona Recognition
- **Zero Configuration**: Handles new personas without code changes
- **Pattern-Based Intelligence**: Automatically detects persona clusters and preferences
- **Smart Keyword Extraction**: Generates expertise areas from persona strings
- **Adaptive Preferences**: Creates custom content priorities dynamically

### Proven Results
```json
{
  "Travel Planner": "Planning 4-day trips → Coastal activities, cuisine, tips",
  "HR Professional": "Compliance forms → Procedures, policies, e-signatures", 
  "Food Contractor": "Vegetarian catering → Recipes, dietary options, planning"
}
```

## 🤖 Models & Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **PyMuPDF** | 1.26.3 | PDF parsing and text extraction |
| **TinyBERT** | 4L/312D | Content classification (156MB) |
| **Sentence Transformers** | 5.0.0 | Semantic similarity (~400MB) |
| **NLTK** | 3.9.1 | Text preprocessing (~200MB) |
| **scikit-learn** | 1.7.0 | ML algorithms and clustering |

**Total Model Size**: ~756MB (within 1GB constraint)

## 🐳 How to Build and Run

### Quick Start (Docker)
```bash
# 1. Build the container
docker build -t adobe-1b .

# 2. Run with input data
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output adobe-1b

# 3. Check results
cat output/challenge1b_output.json
```

### Local Development Setup
```bash
# 1. Install Python 3.10+ and dependencies
pip install -r requirements.txt

# 2. Setup models (first time only)
python scripts/setup_models_1b.py

# 3. Run processing locally - Travel Planner
python main_1b.py --input "input\1\PDFs" --output "output" --test-case "travel_planner_case.json"

# 4. Run processing locally - HR Professional  
python main_1b.py --input "input\2\PDFs" --output "output" --test-case "hr_professional_case.json"

# 5. Run processing locally - Food Contractor
python main_1b.py --input "input\3\PDFs" --output "output"

# 6. Check results
cat output/challenge1b_output.json
```

### Windows Local Setup
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup models
python scripts\setup_models_1b.py

# 3. Run processing - Travel Planner
python main_1b.py --input "input\1\PDFs" --output "output" --test-case "travel_planner_case.json"

# 4. Run processing - HR Professional
python main_1b.py --input "input\2\PDFs" --output "output" --test-case "hr_professional_case.json"

# 5. Run processing - Food Contractor
python main_1b.py --input "input\3\PDFs" --output "output"

# 6. View results
Get-Content .\output\challenge1b_output.json
```

### Input Directory Structure
```
input/
├── 1/PDFs/          # Collection 1: Travel Planning documents + travel_planner_case.json
├── 2/PDFs/          # Collection 2: HR/Adobe Acrobat docs + hr_professional_case.json  
└── 3/PDFs/          # Collection 3: Food/Recipe documents + food_contractor_case.json
```

### Expected Output Format (Sample Compliant)
```json
{
  "metadata": {
    "input_documents": ["South of France - Cities.pdf", "..."],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends.",
    "processing_timestamp": "2025-07-28T16:59:10.382414"
  },
  "extracted_sections": [
    {
      "document": "South of France - Things to Do.pdf",
      "section_title": "Culinary Delights", 
      "importance_rank": 1,
      "page_number": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "South of France - Tips and Tricks.pdf",
      "refined_text": "Planning a trip to the South of France requires thoughtful preparation...",
      "page_number": 1
    }
  ]
}
```

## 📊 Performance Compliance

| Constraint | Limit | Our Implementation | Status |
|------------|-------|-------------------|--------|
| **Processing Time** | ≤60 seconds | ~11-53 seconds | ✅ |
| **Model Size** | ≤1GB | ~756MB | ✅ |
| **Memory Usage** | ≤8GB RAM | ~5-7GB peak | ✅ |
| **Operation Mode** | Offline/CPU | Fully cached | ✅ |
| **Persona Flexibility** | N/A | Any persona supported | ✅ |
| **Output Format** | Adobe samples | Exact match | ✅ |

## 🎯 Proven Test Results

### Collection 1: Travel Planner
```bash
python main_1b.py --input "input\1\PDFs" --output "output" --test-case "travel_planner_case.json"
# ✅ Processing completed in 11.52 seconds
# ✅ Persona: "Travel Planner" preserved
# ✅ Job: "Plan a trip of 4 days for a group of 10 college friends"
# ✅ Top results: Coastal Adventures, Culinary Experiences, Nightlife
```

### Collection 2: HR Professional  
```bash
python main_1b.py --input "input\2\PDFs" --output "output" --test-case "hr_professional_case.json"
# ✅ Processing completed in 53.03 seconds
# ✅ Persona: "HR professional" preserved
# ✅ Job: "Create and manage fillable forms for onboarding and compliance"
# ✅ Top results: Fill and Sign, Request e-signatures, Form creation
```

### Collection 3: Food Contractor
```bash
python main_1b.py --input "input\3\PDFs" --output "output"
# ✅ Processing completed in 29.54 seconds  
# ✅ Persona: "Food Contractor" preserved
# ✅ Job: "Prepare a vegetarian buffet-style dinner menu for corporate gathering"
# ✅ Top results: Vegetarian dishes, Buffet-suitable recipes, Dietary options
```

## 🔧 Troubleshooting

### Docker Validation
```bash
# Test build
docker build -t adobe-1b .

# Test with sample data
mkdir -p test_input/1/PDFs test_output
cp sample.pdf test_input/1/PDFs/
docker run -v $(pwd)/test_input:/app/input -v $(pwd)/test_output:/app/output adobe-1b
```

### Local Development Issues
```bash
# Install missing dependencies
pip install -r requirements.txt

# Fix import errors - check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Test configuration loading
python -c "import yaml; config = yaml.safe_load(open('configs/model_config_1b.yaml', 'r', encoding='utf-8')); print('✅ Config loaded')"

# Test PDF processing
python -c "from src.pdf_intelligence.parser import PDFOutlineExtractor; print('✅ PDF modules loaded')"

# Test persona intelligence
python -c "from src.persona_intelligence.document_analyzer import DocumentAnalyzer; print('✅ Persona modules loaded')"
```

### Performance Issues
```bash
# Docker: Increase memory if needed
docker run --memory=8g adobe-1b

# Docker: Optimize CPU usage  
docker run -e OMP_NUM_THREADS=8 adobe-1b

# Local: Set environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python main_1b.py --input ./input/1
```

### Common Local Issues
| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"` |
| `FileNotFoundError: config` | Ensure you're in project root directory |
| `yaml.constructor.ConstructorError` | Update PyYAML: `pip install PyYAML>=6.0` |
| Slow inference | Reduce batch size in `configs/model_config_1b.yaml` |
| Memory errors | Close other applications, reduce model cache |

## 📁 Project Structure

```
1B/
├── Dockerfile                    # Container with 1GB model support
├── main_1b.py                    # Entry point
├── requirements.txt              # Dependencies
├── approach_explanation.md       # Methodology (300-500 words)
├── configs/
│   └── model_config_1b.yaml     # Unified configuration (1A + 1B)
├── src/
│   ├── pdf_intelligence/        # Enhanced 1A foundation
│   └── persona_intelligence/    # New 1B modules
├── scripts/
│   ├── setup_models_1a.py      # 1A model setup
│   ├── setup_models_1b.py      # 1B model setup
│   └── benchmark_1b.py         # Performance testing
├── input/                       # Input collections (mount point)
├── output/                      # Results (mount point)
└── tests/                       # Unit tests
```

## 🛠️ Development Workflow

### First Time Setup
```bash
# 1. Clone and navigate to project
cd Adobe/1B

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup models and validate
python scripts/setup_models_1b.py

# 4. Test with Travel Planner
python main_1b.py --input "input\1\PDFs" --output "output" --test-case "travel_planner_case.json" --debug

# 5. Test with HR Professional
python main_1b.py --input "input\2\PDFs" --output "output" --test-case "hr_professional_case.json"

# 6. Test with Food Contractor  
python main_1b.py --input "input\3\PDFs" --output "output"
```

### Testing and Validation
```bash
# Run unit tests
python -m pytest tests/ -v

# Test complete pipeline
python tests/test_complete_1b.py

# Benchmark performance
python scripts/benchmark_1b.py

# Test dynamic persona system
python -c "
from src.persona_intelligence.persona_manager import PersonaManager
pm = PersonaManager()
print('✅ Testing Travel Planner...')
profile = pm.parse_persona('Travel Planner')
print(f'Cluster: {profile.persona_cluster.value}')
print(f'Keywords: {profile.keywords[:5]}')
print('✅ Testing Food Contractor...')  
profile = pm.parse_persona('Food Contractor')
print(f'Cluster: {profile.persona_cluster.value}')
print(f'Keywords: {profile.keywords[:5]}')
print('✅ Dynamic persona system working!')
"
```

### Making Changes
```bash
# Edit configuration
nano configs/model_config_1b.yaml

# Test persona changes
python main_1b.py --input "input\1\PDFs" --debug

# Test different personas (system adapts automatically)
echo '{"persona": "Wedding Photographer", "job_to_be_done": "Create a photo timeline"}' > test_case.json
python main_1b.py --input "input\1\PDFs" --test-case "test_case.json"

# Rebuild Docker with changes
docker build -t adobe-1b .
```

## 🏆 Hackathon Submission

- **Challenge**: 1B - Dynamic Persona-Driven Document Intelligence
- **Constraints**: ≤60s, ≤1GB models, ≤8GB RAM, 3-15 documents, Offline
- **Innovation**: Zero-configuration dynamic persona recognition with pattern-based intelligence
- **Models**: TinyBERT (156MB) + Sentence Transformers (400MB) + NLTK (200MB)

### Revolutionary Features
- ✅ **Zero Hardcoding**: Handles any persona without code changes
- ✅ **Pattern Recognition**: Intelligent clustering and keyword extraction  
- ✅ **Sample Compliance**: Exact output format matching Adobe specifications
- ✅ **Proven Results**: Successfully tested with Travel Planner, HR Professional, Food Contractor

### Deliverables
- ✅ **Working Dockerfile** with cached models
- ✅ **approach_explanation.md** detailing dynamic methodology
- ✅ **Complete source code** with revolutionary persona system
- ✅ **Sample outputs** demonstrating adaptive intelligence

### Technical Achievements
- **Dynamic Persona System**: Replaces rigid templates with intelligent pattern recognition
- **Adaptive Job Analysis**: Extracts requirements from any job description
- **Format Compliance**: Produces output exactly matching sample structure
- **Performance Excellence**: Processes 3-15 documents within time constraints

---

<div align="center">
  <p><strong>Built for Adobe India Hackathon 2025 🧠</strong></p>
  <p>🚀 Intelligent • 📊 Multi-Document • 🎯 Persona-Aware • 🐳 Containerized</p>
</div>