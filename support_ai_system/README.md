# SupportMind AI - Self-Learning Support Knowledge Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fault-tolerant, resumable, self-learning AI knowledge platform using local LLMs for enterprise customer support.

## 🎯 Project Overview

SupportMind AI is an enterprise-grade intelligence system that:

- **Ingests** multi-sheet Excel enterprise data
- **Stores** structured knowledge in SQLite with full provenance
- **Retrieves** using hybrid semantic + keyword search
- **Generates** KB articles automatically with AI
- **Learns** continuously from new data
- **Evaluates** itself with comprehensive metrics
- **Recovers** from crashes with checkpointing

## 🏗️ Architecture

```
Excel → Ingestion → SQLite → Embeddings → Hybrid Retrieval
                           ↓
                     AI Orchestrator
                           ↓
                 Learning + Evaluation Loop
                           ↓
                    Version Store + Logs
```

### System Components

| Component | Description |
|-----------|-------------|
| **Ingestion** | Excel parser with schema validation |
| **Storage** | SQLite database with 12+ tables |
| **Retrieval** | FAISS + BM25 hybrid with LLM reranking |
| **Agents** | 6 specialized AI agents |
| **Learning** | Batch training with rolling buffer |
| **Evaluation** | Hit@K, MRR, coverage, quality metrics |
| **Governance** | PII detection, compliance, citations |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Ollama installed and running
- 8GB+ RAM recommended

### Installation

```bash
# Clone or download the project
cd support_ai_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull model
ollama pull llama3
```

### Initialize the System

```bash
# Create database, indexes, and required structures
python main.py --init
```

### Run Baseline Evaluation

```bash
# Evaluate current state without training
python main.py --baseline
```

### Start Training

```bash
# Run full training pipeline
python main.py --train

# Or specify rounds
python main.py --train --rounds 5
```

### Resume After Crash

```bash
# Resume from last checkpoint
python main.py --resume
```

### Full Evaluation

```bash
# Run comprehensive evaluation
python main.py --evaluate --full
```

### Start API Server

```bash
# Start REST API server
python main.py --serve --port 8080
```

## 📁 Directory Structure

```
support_ai_system/
├── data/
│   ├── raw_excel/          # Input Excel files
│   ├── processed/          # Processed data
│   └── backups/            # Database backups
├── db/
│   └── support.db          # SQLite database
├── embeddings/
│   ├── faiss_index/        # Semantic search index
│   └── bm25_index/         # Keyword search index
├── versions/
│   └── kb_versions/        # KB article versions
├── learning_state/
│   ├── checkpoints.json    # Training checkpoints
│   ├── buffer.pkl          # Rolling buffer
│   └── optimizer_state.json
├── logs/
│   ├── system.log          # Main system log
│   ├── learning.log        # Learning-specific log
│   └── errors.log          # Error log
├── metrics/
│   ├── batch_results.csv   # Batch evaluation results
│   ├── trends.json         # Metric trends
│   ├── progress.json       # Current progress
│   └── plots/              # Visualization plots
├── prompts/
│   ├── extractor.txt       # Fact extraction prompt
│   ├── generator.txt       # KB generation prompt
│   ├── validator.txt       # Validation prompt
│   └── scorer.txt          # QA scoring prompt
├── src/
│   ├── ingestion/          # Data ingestion modules
│   ├── storage/            # Database layer
│   ├── retrieval/          # Hybrid retrieval engine
│   ├── agents/             # AI agent implementations
│   ├── learning/           # Learning system
│   ├── evaluation/         # Evaluation metrics
│   ├── governance/         # Compliance & safety
│   └── orchestration/      # System orchestrator
├── main.py                 # CLI entry point
├── config.yaml             # System configuration
└── requirements.txt        # Dependencies
```

## 🗄️ Database Schema

### Core Tables

| Table | Description |
|-------|-------------|
| `conversations` | Customer support conversations |
| `tickets` | Support tickets with metadata |
| `scripts` | Support scripts/runbooks |
| `knowledge_articles` | Generated KB articles |
| `kb_versions` | Article version history |
| `kb_lineage` | Article provenance tracking |
| `questions` | Ground truth Q&A pairs |
| `placeholders` | Template placeholders |
| `qa_scores` | Quality assessment scores |
| `learning_events` | Learning system events |
| `evaluation_runs` | Evaluation run records |
| `learning_checkpoints` | Training checkpoints |

## 🔍 Hybrid Retrieval Pipeline

### 5-Stage Pipeline

```
Query → Semantic Search (FAISS)
     → Keyword Search (BM25)
     → Metadata Filtering
     → LLM Re-ranking
     → Score Fusion
     → Top-K Results
```

### Score Fusion Formula

```
final_score = 0.35 × semantic
            + 0.25 × bm25
            + 0.20 × validation
            + 0.20 × recency
```

## 🤖 Multi-Agent System

### Agents

1. **Extractor Agent** - Extracts structured facts from raw data
2. **Gap Detection Agent** - Identifies missing knowledge
3. **KB Generator Agent** - Creates structured KB articles
4. **Compliance Agent** - Enforces policy compliance
5. **QA Agent** - Scores article quality
6. **Learning Agent** - Optimizes the system

### Knowledge Generation Pipeline

```
Resolve Ticket → Extract → Generate → Validate → QA → Version → Approve → Store → Re-index
```

## 📊 Evaluation Metrics

### Retrieval Metrics
- **Hit@1, Hit@3, Hit@5, Hit@10** - Answer found in top-K
- **MRR** - Mean Reciprocal Rank
- **Coverage** - % questions answerable

### KB Quality Metrics
- **Cosine Similarity** - Semantic similarity to source
- **Structural Score** - Article structure quality
- **LLM Judge Score** - AI quality assessment

### QA Metrics
- **Mean Score** - Average quality score
- **Violation Count** - Policy violations detected

## 🔄 Continuous Learning

### Rolling Buffer System

```python
buffer_size = 500  # Configurable
```

When buffer fills:
1. Re-evaluate current performance
2. Optimize prompts based on feedback
3. Update few-shot examples
4. Refresh search indexes
5. Save checkpoint

### Batch Training

```
Dataset → Split (70/30) → Rounds
                            ↓
                     Batch → Generate → Evaluate → Learn → Save
```

## 💾 Checkpointing & Recovery

### Checkpoint Contents

```json
{
  "batch_id": "batch_042",
  "round": 2,
  "model_version": "llama3",
  "prompt_version": "v1.2",
  "metrics": {...},
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Auto-Resume

On restart, the system automatically:
1. Detects incomplete training
2. Loads last valid checkpoint
3. Resumes from exact position
4. Continues learning

## 🔐 Governance & Safety

### Features

- **Mandatory Citations** - All KB articles cite sources
- **PII Scrubbing** - Automatic PII detection and redaction
- **Confidence Thresholds** - Configurable quality gates
- **Human Override** - Approval queue for low-confidence items
- **Hallucination Detection** - AI-powered fact checking

### Confidence Thresholds

```yaml
auto_approve_threshold: 0.95
human_review_threshold: 0.7
reject_threshold: 0.5
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
llm:
  provider: "ollama"
  ollama:
    endpoint: "http://localhost:11434"
    model: "llama3"

learning:
  buffer_size: 500
  num_rounds: 3
  train_ratio: 0.7

retrieval:
  semantic_top_k: 50
  bm25_top_k: 50
  weights:
    semantic: 0.35
    bm25: 0.25
    validation: 0.20
    recency: 0.20
```

## 📈 Progress Tracking

Real-time progress is displayed:

```
┌────────────────────────────────────────────┐
│ TRAINING PROGRESS                          │
├────────────────────────────────────────────┤
│ Dataset:     ████████░░░░░░░░ 52.3%       │
│ KB Coverage: ██████░░░░░░░░░░ 38.1%       │
│ Evaluation:  ████████████░░░░ 75.0%       │
│ ETA:         12 minutes remaining          │
│ Phase:       Round 2/3 - Batch 15/25      │
└────────────────────────────────────────────┘
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test
pytest tests/test_retrieval.py -v
```

## 📝 Logging

Logs are written to:
- `logs/system.log` - All system events
- `logs/learning.log` - Learning-specific events
- `logs/errors.log` - Errors only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙋 Support

For questions about the dataset, contact Arun at (972) 310 9556 via WhatsApp.

---

Built with ❤️ for the Hack-Nation Global AI Hackathon
