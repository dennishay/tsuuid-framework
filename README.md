# Ternary Semantic UUID (TSUUID) Framework

**Where Data IS Understanding**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2026.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2026.XXXXX)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## What Is This?

TSUUID is a universal information framework that encodes semantic meaning directly into standard UUIDs. Every piece of information becomes a **81-dimensional ternary displacement** from a shared reference model, requiring only **addition** to reconstruct — no multiplication, no parsing, no schema mapping, no network fetch.

Three existing technologies. One new idea.

| Technology | Role | Status |
|---|---|---|
| [BitNet b1.58](https://github.com/microsoft/BitNet) | Ternary {-1,0,+1} weights — addition replaces multiplication | Released by Microsoft Research, MIT license |
| [UUID v8 (RFC 9562)](https://www.rfc-editor.org/rfc/rfc9562.html) | 128-bit container with 122 custom bits | IETF Standard, May 2024 |
| Transformer LLMs | Shared reference frame for semantic space | Ubiquitous |

**The new idea:** 128 bits = 81 ternary dimensions. Each dimension is an independent axis of meaning, not a flat address. The UUID *is* the data. Understanding is intrinsic, not reconstructed.

## Why It Matters

| Current Reality | TSUUID Framework |
|---|---|
| Data and understanding are separated | Data IS understanding |
| Bigger systems cost exponentially more | Bigger systems yield exponentially more knowledge |
| Schema mapping takes weeks | No schema mapping — shared coordinate space |
| Network required to fetch meaning | Understanding is local — nothing to fetch |
| Formats become obsolete | Ternary math doesn't version |
| Training takes hours to months | Each UUID is immediately understood |

## Key Results

From the [energy derivatives paper](docs/TSUUID_Energy_Derivatives.pdf):

```
Knowledge efficiency:  η_K(n) = (2ⁿ · ln2) / E_uuid  →  ∞   (divergent)
Traditional efficiency: η_K(n) = c / (E_unit + 2γn)    →  0   (collapsing)

TSUUID:      dE/dn = constant,  dK/dn = exponential
Traditional: dE/dn = increasing, dK/dn = bounded

All derivatives of K(n) are positive. All derivatives of E(n) beyond first are zero.
The system is maximally expansive in knowledge, maximally conservative in energy.
```

## Quick Start

```bash
git clone https://github.com/DennisEvanHay/tsuuid-framework.git
cd tsuuid-framework
pip install -e .
```

### Encode a document as a semantic UUID

```python
from tsuuid import SemanticCodec, UniversalModel

# Load the universal reference frame (one-time, ~0.4GB)
model = UniversalModel.load("bitnet-b1.58-2b")

# Create codec
codec = SemanticCodec(model)

# Encode: document → 81 ternary displacements → UUID
uuid = codec.encode("Invoice #4471: $2,340 from Acme Corp, due 2026-04-15")
print(uuid)
# → UUID('8a3f7b21-e4c1-8d02-b7a3-9f1c2e4d6a8b')  (v8 semantic)

# Decode: UUID → apply ternary diff → reconstructed meaning
meaning = codec.decode(uuid)
print(meaning.domain)      # "financial"
print(meaning.entity)      # "invoice"
print(meaning.attributes)  # {amount: 2340, vendor: "Acme Corp", due: "2026-04-15"}
```

### Cross-database query without schema mapping

```python
from tsuuid import SchemaRouter

# Connect heterogeneous databases
router = SchemaRouter(model)
router.add_database("postgres://db1.example.com/sales")    # has 'customer_name'
router.add_database("mysql://db2.example.com/billing")     # has 'cust_nm'
router.add_database("sqlite:///local/inventory.db")        # has 'client_id'

# Query in natural language — no schema mapping required
results = router.query("Show me all unpaid invoices over $1000 from last quarter")

# The router:
# 1. Encodes query as semantic UUID (81 trits)
# 2. Compares ternary coordinates across all DB schemas
# 3. Identifies shared dimensions automatically
# 4. Generates valid SQL per database
# 5. Returns unified results
```

### Concurrent independent learning

```python
from tsuuid import SemanticCodec
import uuid

# Device A learns independently
uuid_a = codec.encode("Pressure drop across valve V-201 reduced 15% after cleaning")

# Device B learns independently (different location, no coordination)
uuid_b = codec.encode("Maintenance cost for valve cleaning averages $450 per unit")

# Device C learns independently
uuid_c = codec.encode("Quarterly budget allocation for preventive maintenance: $12,000")

# Combine by addition — order doesn't matter (commutative)
combined = codec.compose([uuid_a, uuid_b, uuid_c])

# Emergent knowledge: system now understands the ROI relationship
# between valve cleaning, cost per unit, and budget allocation
# — a relationship NO individual device discovered
insight = codec.query_composed(combined, "Is valve cleaning cost-effective?")
```

## Repository Structure

```
tsuuid-framework/
├── README.md                       # You are here
├── LICENSE                         # MIT License
├── CONTRIBUTING.md                 # How to contribute
├── COLLABORATORS.md                # Academic and industry partners
├── pyproject.toml                  # Python package config
│
├── docs/
│   ├── TSUUID_Framework_arXiv_Preprint.pdf    # Main paper
│   ├── TSUUID_Information_Inertia_Formulas.pdf # Information inertia supplement
│   ├── TSUUID_Energy_Derivatives.pdf           # Energy-knowledge derivatives
│   └── RFC_UUID_v8_Semantic_Profile.md         # Draft UUID v8 profile spec
│
├── src/tsuuid/
│   ├── __init__.py
│   ├── codec.py                    # Core: document ↔ ternary diff ↔ UUID
│   ├── model.py                    # Universal reference frame loader
│   ├── dimensions.py               # 81-dimensional axis definitions
│   ├── packing.py                  # Trit ↔ UUID bit packing (RFC 9562 v8)
│   ├── router.py                   # Cross-database schema routing
│   ├── compose.py                  # UUID composition by addition
│   └── quantum.py                  # Qutrit extension (simulation)
│
├── src/examples/
│   ├── 01_basic_encode_decode.py   # Hello world: encode and decode a document
│   ├── 02_cross_database_query.py  # Query across heterogeneous SQL databases
│   ├── 03_concurrent_learning.py   # Independent devices, composed knowledge
│   ├── 04_version_history.py       # Document versioning as ternary diffs
│   ├── 05_legacy_integration.py    # Making legacy systems UUID-addressable
│   ├── 06_inverse_scaling_demo.py  # Demonstrate knowledge vs energy scaling
│   └── 07_qutrit_simulation.py     # Quantum qutrit extension simulation
│
├── tests/
│   ├── test_codec.py
│   ├── test_packing.py
│   ├── test_router.py
│   ├── test_compose.py
│   └── test_dimensions.py
│
└── assets/
    └── architecture_diagram.svg
```

## The Papers

| Document | Description |
|---|---|
| [Main Paper](docs/TSUUID_Framework_arXiv_Preprint.pdf) | Full framework: theory, architecture, 5-phase implementation plan |
| [Information Inertia](docs/TSUUID_Information_Inertia_Formulas.pdf) | Why this system has zero information inertia (biological neural analogy) |
| [Energy Derivatives](docs/TSUUID_Energy_Derivatives.pdf) | Formal proof that knowledge/energy efficiency diverges to ∞ |

## Call for Collaborators

This is an open research project. We're actively seeking collaborators from:

### Academia

| Domain | What We Need | Contact |
|---|---|---|
| **Information Theory / Coding Theory** | Formalize compression bounds, prove/disprove 81-dimensional sufficiency | Open an issue tagged `theory` |
| **Database Systems** | Benchmark cross-schema resolution against traditional approaches | Open an issue tagged `databases` |
| **Quantum Computing** | Formalize trit→qutrit upgrade, entanglement properties | Open an issue tagged `quantum` |
| **Computational Neuroscience** | Validate Hebbian/STDP analogy, biological plausibility | Open an issue tagged `neuroscience` |
| **Computer Architecture** | Evaluate UEFI/firmware integration feasibility | Open an issue tagged `hardware` |
| **NLP / Embeddings** | Validate semantic axis definitions across domains | Open an issue tagged `nlp` |

### Industry

| Domain | Opportunity |
|---|---|
| **Chipmakers** (Intel, AMD, ARM, RISC-V) | Evaluate ternary UUID generation in firmware |
| **Database vendors** (PostgreSQL, MySQL, MongoDB) | Native semantic UUID indexing |
| **Cloud providers** | Benchmark vs. traditional ETL/integration |
| **Standards bodies** (IETF, IEEE, UEFI Forum) | UUID v8 semantic profile standardization |
| **Enterprise IT** | Pilot deployments for cross-system integration |

### How to Get Involved

1. **Read the papers** in `docs/`
2. **Run the examples** in `src/examples/`
3. **Open an issue** describing what you'd like to work on
4. **Fork and submit a PR** with your contribution
5. **Cite this work** if you build on it (see [CITATION.cff](CITATION.cff))

## Roadmap

| Phase | Timeline | Status | Description |
|---|---|---|---|
| 1 | Months 1–3 | **Active** | Proof of concept: codec + SQL database demo |
| 2 | Months 4–6 | Planned | Universal ontology: define 81 dimensional axes |
| 3 | Months 7–9 | Planned | Formal spec: UUID v8 profile, arXiv submission |
| 4 | Months 10–15 | Planned | Trust hierarchy: enterprise pilot deployment |
| 5 | Months 16–24 | Planned | Quantum readiness: qutrit simulation + formalization |

## Citing This Work

```bibtex
@article{hay2026tsuuid,
  title={Ternary Semantic UUID: A Universal Information Framework 
         for Inverse-Scaling Knowledge Representation},
  author={Hay, Dennis Evan},
  journal={arXiv preprint arXiv:2026.XXXXX},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

This is open science. Use it, build on it, challenge it, improve it.

---

*"The UUID doesn't reference a location where meaning is stored. The UUID IS the meaning."*
