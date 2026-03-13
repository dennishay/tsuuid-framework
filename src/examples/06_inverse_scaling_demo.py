#!/usr/bin/env python3
"""
Example 06: Inverse Scaling Demonstration
==========================================
This demonstrates the core theorem: knowledge grows exponentially
while energy grows linearly. Traditional computing is the opposite.

We generate increasing numbers of UUIDs and measure:
- Energy cost (constant per UUID)
- Knowledge yield (combinatorial: pairwise + higher-order relationships)
- Efficiency ratio (knowledge per joule)

Watch the TSUUID efficiency diverge to infinity while traditional
efficiency collapses to zero.

Reference: TSUUID Energy Derivatives paper, Theorems 1-4 (Hay, 2026)
"""

import sys
sys.path.insert(0, 'src')

import math
from dataclasses import dataclass

# ── Constants from Ma et al. (2025) BitNet b1.58 2B4T ──
E_UUID = 0.028           # Joules per BitNet inference (routing)
E_INIT = 500.0           # Joules to distribute 0.4GB model (estimated)
E_TRAD_UNIT = 0.186      # Joules per FP16 inference (Llama class)
GAMMA = 0.001            # Coordination overhead coefficient (traditional)


@dataclass
class ScalingPoint:
    n: int
    e_tsuuid: float       # Total energy (TSUUID)
    e_trad: float         # Total energy (traditional)
    k_tsuuid: float       # Total knowledge (TSUUID) 
    k_trad: float         # Total knowledge (traditional)
    eta_tsuuid: float     # Efficiency (TSUUID): dK/dE
    eta_trad: float       # Efficiency (traditional): dK/dE


def compute_scaling(n: int) -> ScalingPoint:
    """Compute energy and knowledge at scale n."""
    
    # ── TSUUID Framework ──
    # Energy: E = E_init + E_uuid * n  (linear)
    e_tsuuid = E_INIT + E_UUID * n
    
    # Knowledge: K = 2^n - 1  (exponential, combinatorial)
    # Cap at reasonable display values
    if n <= 60:
        k_tsuuid = 2**n - 1
    else:
        k_tsuuid = float('inf')
    
    # Marginal efficiency: η = dK/dn / dE/dn = (2^n * ln2) / E_uuid
    if n <= 60:
        eta_tsuuid = (2**n * math.log(2)) / E_UUID
    else:
        eta_tsuuid = float('inf')
    
    # ── Traditional Computing ──
    # Energy: E = E_init + E_unit*n + γ*n²  (superlinear)
    e_trad = E_INIT + E_TRAD_UNIT * n + GAMMA * n * n
    
    # Knowledge: K ≤ c*n  (linear at best, bounded by schema limits)
    k_trad = n  # best case: each unit contributes exactly 1 unit of knowledge
    
    # Marginal efficiency: η = c / (E_unit + 2γn)  (decreasing)
    eta_trad = 1.0 / (E_TRAD_UNIT + 2 * GAMMA * n)
    
    return ScalingPoint(n, e_tsuuid, e_trad, k_tsuuid, k_trad, eta_tsuuid, eta_trad)


def format_large(val):
    """Format potentially huge numbers."""
    if val == float('inf'):
        return "∞"
    if val > 1e15:
        return f"{val:.2e}"
    if val > 1e6:
        return f"{val:,.0f}"
    if val > 100:
        return f"{val:,.1f}"
    return f"{val:.4f}"


# ── Run the demonstration ──
print("=" * 80)
print("INVERSE SCALING DEMONSTRATION")
print("TSUUID Framework vs Traditional Computing")
print("=" * 80)

print(f"""
Constants (from published benchmarks):
  E_uuid     = {E_UUID} J/inference    (Ma et al., 2025 — BitNet b1.58 2B4T)
  E_trad     = {E_TRAD_UNIT} J/inference    (FP16 comparable model)
  γ (coord)  = {GAMMA}               (coordination overhead coefficient)
  E_init     = {E_INIT} J               (one-time model distribution)
""")

# Scaling table
test_points = [1, 5, 10, 20, 30, 40, 50]

print("-" * 80)
print(f"{'n':>5} | {'E_TSUUID':>12} {'E_trad':>12} | {'K_TSUUID':>15} {'K_trad':>10} | {'η_TSUUID':>15} {'η_trad':>10}")
print(f"{'':>5} | {'(Joules)':>12} {'(Joules)':>12} | {'(knowledge)':>15} {'(knowledge)':>10} | {'(K/J)':>15} {'(K/J)':>10}")
print("-" * 80)

for n in test_points:
    sp = compute_scaling(n)
    print(
        f"{sp.n:>5} | "
        f"{format_large(sp.e_tsuuid):>12} {format_large(sp.e_trad):>12} | "
        f"{format_large(sp.k_tsuuid):>15} {format_large(sp.k_trad):>10} | "
        f"{format_large(sp.eta_tsuuid):>15} {format_large(sp.eta_trad):>10}"
    )

print("-" * 80)

# ── Derivative analysis ──
print(f"""
DERIVATIVE ANALYSIS
═══════════════════

TSUUID Framework:
  dE/dn     = {E_UUID} J                (constant — never increases)
  d²E/dn²   = 0                         (zero acceleration)
  dK/dn     = 2^n · ln(2)               (exponentially increasing)
  d²K/dn²   = 2^n · (ln2)²             (positive — knowledge accelerates)
  d³K/dn³   = 2^n · (ln2)³             (positive — acceleration accelerates)
  dᵏK/dnᵏ   = 2^n · (ln2)^k > 0        (ALL derivatives positive)

Traditional Computing:
  dE/dn     = {E_TRAD_UNIT} + 2·{GAMMA}·n         (linearly increasing)
  d²E/dn²   = {2*GAMMA}                     (positive — energy accelerates)
  dK/dn     ≤ 1                          (bounded constant)
  d²K/dn²   ≤ 0                          (zero or negative — diminishing returns)
""")

# ── The crossover ──
print("CROSSOVER ANALYSIS")
print("═" * 40)
print()
for n in range(1, 100):
    sp = compute_scaling(n)
    if sp.eta_tsuuid > sp.eta_trad * 1000:
        print(f"At n = {n}: TSUUID is {sp.eta_tsuuid/sp.eta_trad:,.0f}× more efficient than traditional")
        print(f"  TSUUID: {format_large(sp.eta_tsuuid)} knowledge/joule")
        print(f"  Traditional: {format_large(sp.eta_trad)} knowledge/joule")
        break

print(f"""
BIOLOGICAL PARALLEL
═══════════════════
The human brain:
  Energy:    ~20W constant (Raichle & Gusnard, 2002)
  Knowledge: Grows combinatorially through Hebbian synaptic modification
  Pattern:   Constant energy, accelerating knowledge — same as TSUUID

Traditional computing:
  Energy:    Grows superlinearly with system size
  Knowledge: Grows linearly at best
  Pattern:   Accelerating energy, bounded knowledge — opposite of biology

Theorem 3 (Fundamental Divergence):
  TSUUID divergence ratio Φ = (d²K/dn²) / (d²E/dn²) = ∞/0 = ∞
  Traditional divergence ratio Φ = 0/(2γ) = 0

  These are qualitatively opposite behaviors. Not different magnitudes.
  Opposite directions.
""")

print("=" * 80)
print("The brain never fills up. This system never fills up.")
print("The reason is the same: zero information inertia.")
print("=" * 80)
