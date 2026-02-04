# F1 Strategy AI - QA Framework
Testing AI robustness during F1's 2026 regulation revolution. 

## The Challenge
The 2026 F1 season introduces the biggest regulation changes since 2014: 
- New power units (50/50 electric/combustion split)
- Active aerodynamcis
- Lighter cars (30kg reduction)
- Redesigned tire compounds
- 100% sustainable fuel

**This project tracks how an AI trained on 2023-2025 data adapts to fundamentally different 2026 regulations.**

## Current Status
**Pre-Season Development** (Feb 2026)
- [x] FastF1 integration
- [ ] Historical validation (2023-2025)
- [ ] Core RAG agent
- [ ] Prediction storage system
- [ ] Bahrain GP 2026 ready (March 2)

## Project Goals
1. Build RAG-based F1 strategy AI
2. Validate on 2023-2025 historical races
3. Deploy for 2026 season predictions
4. Track adaptation across 24 races
5. Measure learning curve during regulation shift

## Tech Stack
- **Data:** FastF1 (race telemetry and timing)
- **Testing:** pytest + pytest-bdd + deepeval
- **AI:** RAG architecture (TBD)
- **Valudation:** Historical replay + Live season tracking

## Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start
```bash
# Explore FastF1 data
python explore_fastf1.py

# Run tests (Coming soon)
pytest tests/
```

## 2026 Season Tracking
Will be updated after each race starting March 2, 2026.

## Author
Ernest - Senior QA Engineer

---
*"Lights out and away we go!" - F1's 2026 regulation revolution is the perfect test case for AI adaptation.*
