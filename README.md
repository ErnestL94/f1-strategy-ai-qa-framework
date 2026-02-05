# F1 Strategy AI - RAG Testing Framework
Testing AI-powered race strategy systems for F1's 2026 regulation revolution.

## The Challenge
Can AI-powered strategy systems adapt to F1's biggest regulation changes since 2014?

**2026 brings:**
- New power units (50/50 electric/combustion)
- Active aerodynamics
- Lighter cars (30kg reduction)
- Redesigned tire compounds
- 100% sustainable fuel

**This project compares three approaches:**
1. **Rule-Based Agent** - Traditional deterministic logic
2. **Pure RAG (Retrieval-Augmented Generation)** - LLM reasoning with vector retrieval
3. **Hybrid RAG** - Explicit rules + flexible AI reasoning

---

## Current Results (v0.4.0 - February 2026)

### System Performance Comparison

| Approach | Accuracy | Test Set | Strengths | Weaknesses |
|----------|----------|----------|-----------|------------|
| **Rule-Based** | 100% (15/15) | Golden scenarios | Deterministic, reliable | Can't learn, rigid |
| **Pure RAG v1** | 53% (8/15) | Golden scenarios | Learns patterns | Unreliable, hallucinations |
| **Hybrid RAG v2** | 80% (12/15) | Golden scenarios | Balanced approach | Still below 100% |

### Test Coverage
```
Unit Tests:        14/14
Rule-Based Agent:  15/15 (100%)
RAG Agent v1:       8/15 (53%)
RAG Agent v2:      12/15 (80%)
Adversarial Tests:  8/8
```

**Code Coverage: 87%**

---

## 📊 What We've Built

### Phase 1: Rule-Based Baseline (v0.1-0.3)
- Pure deterministic agent with FIA compliance
- Input validation and safety guardrails
- 100% accuracy on 15 golden scenarios
- Adversarial testing framework

### Phase 2: RAG Implementation (v0.4 - Current)
- Vector database (ChromaDB) with 15 golden scenarios
- Ollama integration (llama3.2:3b local LLM)
- Pure RAG v1: 53% accuracy (too flexible, unreliable)
- Hybrid RAG v2: 80% accuracy (rules + LLM reasoning)
- Deterministic inference (temperature=0, seed=42)

**Key Achievement:** Improved from 53% → 80% through iterative prompt engineering

### Architecture: Hybrid RAG v2
```python
Decision Flow:
├─ STEP 1: Final Laps (≤3 remaining) → STAY_OUT  [Explicit Rule]
├─ STEP 2: Safety Car + Old Tires → BOX          [Explicit Rule]
├─ STEP 3: Fresh Tires (≤5 laps) → STAY_OUT     [Explicit Rule]
├─ STEP 4: 80%+ Consensus → Follow Pattern       [RAG: Vector Retrieval]
└─ STEP 5: Complex Analysis → LLM Reasoning      [RAG: LLM Analysis]

Coverage:
- Steps 1-3 (Explicit Rules): 10/12 scenarios = 83% accuracy
- Steps 4-5 (RAG Reasoning):   2/3  scenarios = 67% accuracy
```

### Trade-offs Analysis

**Why 80% vs 100%?**

The 3 failing scenarios demonstrate the limits of current approach:
1. **silverstone_2023_hamilton_lap34_sc** - Complex safety car timing
2. **singapore_2023_russell_lap62** - Edge case in final lap detection  
3. **spa_2023_norris_lap17** - Rain scenario (no weather rule)

**Could we hit 100%?** Yes, by adding more explicit rules. But that defeats the purpose of RAG - we'd just rebuild the rule-based system with extra steps.

**The hybrid balances:**
- Reliability for high-stakes decisions (final laps, safety car)
- Flexibility for complex multi-factor scenarios
- Accepts 80% accuracy to preserve learning capability

---

## Next Phase: ML Training on Real Data

### Planned: FastF1 Integration (v0.5)

**Goal:** Train on thousands of real pit stop decisions from 2023-2025 seasons

**Approach:**
1. **Data Collection** - Extract all pit decisions from FastF1 API
2. **Feature Engineering** - Tire age, degradation, gaps, weather, SC status
3. **Model Training** - Random Forest / XGBoost baseline
4. **Evaluation** - Compare ML vs RAG vs Rules

**Expected Dataset:**
- ~1,000+ pit stop decisions across 2023 season
- Real telemetry (lap times, tire wear, gaps)
- Ground truth labels (actual team decisions)

**Success Metric:** Beat 80% accuracy with data-driven approach

---

## Technical Stack

**AI/ML:**
- **ChromaDB** - Vector database for scenario retrieval
- **Ollama** (llama3.2:3b) - Local LLM inference
- **Sentence Transformers** - Text embeddings (384-dim)
- **FastF1** (planned) - F1 telemetry data

**Testing:**
- **pytest** - Test framework (27 tests passing)
- **Black** - Code formatting
- **Coverage.py** - 87% code coverage

**Development:**
- **Python 3.10+**
- **Pre-commit hooks** - Auto-formatting
- **GitHub Actions** - CI/CD pipeline

---

## Quick Start

### Installation
```bash
# Clone the repository
git clone
cd f1-strategy-ai-qa-framework

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Build Vector Database
```bash
# Ingest golden scenarios into vector DB
python scripts/ingest_golden_scenarios.py
```

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/unit/ -v           # Unit tests (14 tests)
pytest tests/evaluation/ -v     # Golden dataset validation (5 tests)
pytest tests/adversarial/ -v    # Security & edge cases (8 tests)

### Explore Data
```bash
# Explore FastF1 data for Silverstone 2023
python explore_fastf1.py
```

### View Dashboard
```bash
# Start local server
python -m http.server 8000

# Open browser to:
# http://localhost:8000/dashboard/
```

The dashboard shows:
- Test suite results (unit, evaluation, adversarial)
- Code coverage metrics
- Golden dataset accuracy by track
- Agent capabilities

---

## Project Structure
```
f1-strategy-ai-qa-framework/
├── config/
│   └── pytest.ini                    # Test configuration
│
├── data/
│   └── cache/                        # FastF1 cached race data
│
├── datasets/
│   ├── golden/                       # Real race scenarios from actual GPs
│   │   └── silverstone_2023_scenarios.json
│   └── adversarial/                  # Edge cases & attack scenarios
│       └── impossible_scenarios.json
│
├── src/
│   ├── rag/
│   │   ├── agent.py                  # F1StrategyAgent - core decision engine
│   │   └── __init__.py
│   └── validators/
│       ├── input_validator.py        # Input validation & guardrails
│       ├── schema_validator.py       # JSON schema validation
│       └── __init__.py
│
├── tests/
│   ├── conftest.py                   # Shared pytest fixtures
│   ├── unit/
│   │   └── test_golden_dataset.py    # Dataset structure validation
│   ├── evaluation/
│   │   └── test_strategy_agent.py    # Agent performance tests
│   └── adversarial/
│       └── test_hallucination_guards.py  # Security & robustness tests
│
├── explore_fastf1.py                 # Data exploration script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Results Deep Dive

### Golden Dataset Scenarios

**Silverstone 2023 British GP** (5/5 for rules, 4/5 for RAG):
- [Pass] Norris Lap 1 - Fresh tire handling
- [Pass] Russell Lap 28 - Extended stint strategy
- [Pass] Piastri Lap 30 - Pre-SC pit timing
- [Pass] Norris Lap 34 - Restart defense
- [Fail] Hamilton Lap 34 - Safety car opportunity (RAG failed)

**Singapore 2023 GP** (5/5 for rules, 4/5 for RAG):
- [Pass] Leclerc Lap 1 - Split strategy execution
- [Pass] Sainz Lap 20 - Safety car transition
- [Pass] Russell Lap 44 - VSC attack timing
- [Pass] Sainz Lap 60 - Final laps management
- [Fail] Russell Lap 62 - Final lap edge case (RAG failed)

**Spa 2023 Belgian GP** (5/5 for rules, 4/5 for RAG):
- [Pass] Verstappen Lap 14 - Fresh soft management
- [Fail] Norris Lap 17 - Rain gamble (RAG failed)
- [Pass] Russell Lap 22 - Medium degradation
- [Pass] Leclerc Lap 24 - Overcut execution
- [Pass] Hamilton Lap 42 - Fastest lap opportunity

### Failure Analysis

**Why RAG fails:**
1. **Complex weather decisions** - No explicit rain rule
2. **Edge cases** - Lap 62 boundary condition
3. **Multi-factor scenarios** - Safety car timing nuances

**Why we don't just add more rules:**
- Would turn hybrid into pure rule-based system
- Defeats purpose of showing RAG capabilities
- Portfolio value is in demonstrating trade-offs

---

## Lessons Learned

### 1. RAG Requires Constraints
**Finding:** Pure RAG (53%) was too unreliable for production
**Solution:** Added explicit rules for high-stakes scenarios
**Result:** Hybrid RAG (80%) balances reliability + flexibility

### 2. Prompt Engineering Matters
**Iterations:**
- v1: Narrative prompt → 53% accuracy
- v2: Added decision tree → 73% accuracy  
- v2.1: Deterministic inference (temp=0, seed=42) → 80% accuracy

### 3. LLMs Are Non-Deterministic
**Challenge:** Same scenario gave different results across runs
**Solution:** temperature=0 + seed=42 for reproducibility
**Learning:** Even at temp=0, LLMs have variance

### 4. Test-Driven Development Works
**Process:** Red (adversarial tests) → Green (guardrails) → Refactor
**Result:** 25% → 100% test pass rate

---

## Roadmap

### Immediate
- [x] Rule-based baseline (100%)
- [x] RAG v1 implementation (53%)
- [x] Hybrid RAG v2 (80%)
- [x] Deterministic inference

### Short Term
- [ ] FastF1 data collection (2023 season)
- [ ] Feature engineering pipeline
- [ ] ML baseline (Random Forest)
- [ ] Compare ML vs RAG vs Rules

### Medium Term
- [ ] Deploy for Bahrain GP 2026 (March 2)
- [ ] Live prediction tracking
- [ ] Measure adaptation to new regulations

### Long Term
- [ ] Track 24-race season
- [ ] Publish findings on AI adaptation
- [ ] Identify regulation-invariant patterns

---

## Author
**Ernest** - Senior Lead QA Engineer & Technical Product Owner
