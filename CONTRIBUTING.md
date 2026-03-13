# Contributing to TSUUID Framework

Thank you for your interest in contributing. This is an open research project and every contribution — from theoretical critique to working code — advances the framework.

## Ways to Contribute

### 1. Theoretical Critique
The strongest contribution is finding where the theory breaks. If you can demonstrate a case where:
- 81 ternary dimensions are insufficient for a domain
- The inverse scaling property has a practical ceiling
- The biological neural analogy fails under specific conditions
- The compression ratio degrades for certain content types

...open an issue with the tag `theory` and your analysis. Falsification is as valuable as validation.

### 2. Empirical Validation
- Benchmark the codec against traditional schema mapping
- Test cross-database resolution accuracy on real-world schemas
- Measure actual energy consumption on target hardware
- Validate dimensional axis definitions against diverse domains

### 3. Code
- Implement missing components in `src/tsuuid/`
- Add examples demonstrating new use cases
- Write tests (we need coverage)
- Optimize the trit packing algorithm
- Build MCP server integration for Claude Code

### 4. Documentation
- Improve explanations in the papers
- Add tutorials for specific use cases
- Translate papers to other languages
- Create visualizations of the 81-dimensional space

## Development Setup

```bash
git clone https://github.com/DennisEvanHay/tsuuid-framework.git
cd tsuuid-framework
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear commit messages
4. Add or update tests as appropriate
5. Run `pytest` and ensure all tests pass
6. Submit a PR with a clear description of what and why

## Code Style

- Python 3.10+
- Type hints on all public functions
- Docstrings (NumPy style) on all public functions
- No dependencies beyond standard library + numpy for core codec
- Optional dependencies clearly marked

## Academic Contributions

If your contribution results in publishable findings:
- You retain full authorship of your own work
- We ask for citation of the TSUUID framework paper
- Co-authorship on framework papers is offered for substantial theoretical contributions
- All contributors are listed in COLLABORATORS.md

## Issue Labels

| Label | Meaning |
|---|---|
| `theory` | Theoretical analysis, proofs, critiques |
| `databases` | Cross-database resolution, SQL integration |
| `quantum` | Qutrit extension, entanglement formalization |
| `neuroscience` | Biological neural analogy, Hebbian learning |
| `hardware` | UEFI, firmware, silicon integration |
| `nlp` | Semantic axes, embedding validation |
| `codec` | Core encode/decode implementation |
| `good-first-issue` | Accessible entry points for new contributors |

## Code of Conduct

Be respectful, be rigorous, be curious. This project values intellectual honesty above all else. Disagreement is welcome; dismissiveness is not.

## Questions?

Open a Discussion thread. We read everything.
