"""
tsuuid.dimensions — The 81-Dimensional Semantic Axis Definitions

Each dimension represents an independent axis of meaning.
Trit values: -1 (negative pole), 0 (neutral/absent), +1 (positive pole)

IMPORTANT: These axes are preliminary and require empirical validation.
This is where academic collaboration is most needed. The axes should
be validated against diverse domain schemas to ensure universal coverage.

Reference: TSUUID Framework paper, Section 3.2 (Hay, 2026)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Axis:
    """A single semantic dimension."""
    id: int
    name: str
    negative: str   # meaning of trit = -1
    neutral: str    # meaning of trit = 0
    positive: str   # meaning of trit = +1
    layer: str      # trust hierarchy layer this belongs to
    
    def describe_trit(self, value: int) -> str:
        if value == -1:
            return f"{self.name}: {self.negative}"
        elif value == 0:
            return f"{self.name}: {self.neutral}"
        elif value == 1:
            return f"{self.name}: {self.positive}"
        raise ValueError(f"Invalid trit value: {value}")


# ============================================================
# PRELIMINARY AXIS DEFINITIONS
# Organized by trust hierarchy layer (trits 1-81)
# ============================================================

# Layer: Protocol / Standards (trits 1-10)
# These dimensions encode structural/protocol-level semantics
PROTOCOL_AXES = [
    Axis(1,  "temporality",      "past",         "atemporal",     "future",        "protocol"),
    Axis(2,  "modality",         "deletion",      "reference",     "creation",      "protocol"),
    Axis(3,  "certainty",        "denied",        "unknown",       "confirmed",     "protocol"),
    Axis(4,  "cardinality",      "singular",      "unspecified",   "plural",        "protocol"),
    Axis(5,  "obligation",       "forbidden",     "optional",      "required",      "protocol"),
    Axis(6,  "visibility",       "private",       "contextual",    "public",        "protocol"),
    Axis(7,  "mutability",       "immutable",     "unspecified",   "mutable",       "protocol"),
    Axis(8,  "directionality",   "inbound",       "undirected",    "outbound",      "protocol"),
    Axis(9,  "granularity",      "aggregate",     "unspecified",   "atomic",        "protocol"),
    Axis(10, "lifecycle",        "archived",      "active",        "draft",         "protocol"),
]

# Layer: Hardware / Platform (trits 11-20)
HARDWARE_AXES = [
    Axis(11, "compute_class",    "embedded",      "general",       "server",        "hardware"),
    Axis(12, "connectivity",     "offline",       "intermittent",  "always-on",     "hardware"),
    Axis(13, "storage_class",    "ephemeral",     "cached",        "persistent",    "hardware"),
    Axis(14, "precision",        "approximate",   "standard",      "exact",         "hardware"),
    Axis(15, "endianness",       "little",        "unspecified",   "big",           "hardware"),
    Axis(16, "word_width",       "narrow",        "standard",      "wide",          "hardware"),
    Axis(17, "security_level",   "untrusted",     "standard",      "hardened",      "hardware"),
    Axis(18, "power_state",      "sleep",         "idle",          "active",        "hardware"),
    Axis(19, "locality",         "remote",        "networked",     "local",         "hardware"),
    Axis(20, "determinism",      "stochastic",    "unspecified",   "deterministic", "hardware"),
]

# Layer: Organization / Enterprise (trits 21-35)
ORGANIZATION_AXES = [
    Axis(21, "ownership",        "external",      "shared",        "internal",      "organization"),
    Axis(22, "confidentiality",  "classified",    "standard",      "open",          "organization"),
    Axis(23, "authority",        "subordinate",   "peer",          "authoritative", "organization"),
    Axis(24, "compliance",       "non-compliant", "unassessed",    "compliant",     "organization"),
    Axis(25, "cost_center",      "expense",       "neutral",       "revenue",       "organization"),
    Axis(26, "urgency",          "deferred",      "normal",        "critical",      "organization"),
    Axis(27, "approval_state",   "rejected",      "pending",       "approved",      "organization"),
    Axis(28, "department_class", "support",       "cross-functional", "operations", "organization"),
    Axis(29, "geographic_scope", "local",         "regional",      "global",        "organization"),
    Axis(30, "retention",        "delete",        "review",        "retain",        "organization"),
    Axis(31, "audit_status",     "flagged",       "unaudited",     "cleared",       "organization"),
    Axis(32, "stakeholder",      "external",      "mixed",         "internal",      "organization"),
    Axis(33, "maturity",         "experimental",  "standard",      "proven",        "organization"),
    Axis(34, "impact",           "low",           "moderate",      "high",          "organization"),
    Axis(35, "frequency",        "rare",          "periodic",      "continuous",    "organization"),
]

# Layer: Application / Domain (trits 36-55)
APPLICATION_AXES = [
    Axis(36, "domain",           "technical",     "general",       "business",      "application"),
    Axis(37, "data_type",        "unstructured",  "semi-structured", "structured",  "application"),
    Axis(38, "flow_direction",   "outflow",       "static",        "inflow",        "application"),
    Axis(39, "value_sign",       "negative",      "zero",          "positive",      "application"),
    Axis(40, "scale",            "small",         "medium",        "large",         "application"),
    Axis(41, "relationship",     "child",         "standalone",    "parent",        "application"),
    Axis(42, "status",           "failed",        "in-progress",   "complete",      "application"),
    Axis(43, "priority",         "low",           "normal",        "high",          "application"),
    Axis(44, "source_type",      "computed",      "hybrid",        "observed",      "application"),
    Axis(45, "currency",         "stale",         "current",       "projected",     "application"),
    Axis(46, "aggregation",      "detail",        "summary",       "rollup",        "application"),
    Axis(47, "sentiment",        "negative",      "neutral",       "positive",      "application"),
    Axis(48, "action_type",      "query",         "reference",     "mutation",      "application"),
    Axis(49, "entity_class",     "event",         "abstract",      "physical",      "application"),
    Axis(50, "quantity_type",    "rate",           "absolute",      "cumulative",   "application"),
    Axis(51, "time_horizon",     "historical",    "current",       "forecast",      "application"),
    Axis(52, "validation",       "invalid",       "unvalidated",   "validated",     "application"),
    Axis(53, "version_state",    "deprecated",    "current",       "next",          "application"),
    Axis(54, "dependency",       "blocked",       "independent",   "blocking",      "application"),
    Axis(55, "format_class",     "binary",        "mixed",         "text",          "application"),
]

# Layer: Entity / Record (trits 56-70)
ENTITY_AXES = [
    Axis(56, "entity_role",      "object",        "attribute",     "subject",       "entity"),
    Axis(57, "identity_type",    "anonymous",     "pseudonymous",  "identified",    "entity"),
    Axis(58, "completeness",     "partial",       "standard",      "complete",      "entity"),
    Axis(59, "encoding",         "compressed",    "standard",      "expanded",      "entity"),
    Axis(60, "reference_type",   "symbolic",      "direct",        "semantic",      "entity"),
    Axis(61, "nullability",      "null",          "default",       "populated",     "entity"),
    Axis(62, "uniqueness",       "duplicate",     "unverified",    "unique",        "entity"),
    Axis(63, "ordering",         "unordered",     "partial",       "total",         "entity"),
    Axis(64, "indexability",     "unindexed",     "secondary",     "primary",       "entity"),
    Axis(65, "volatility",       "stable",        "moderate",      "volatile",      "entity"),
    Axis(66, "precision_level",  "approximate",   "standard",      "high-precision","entity"),
    Axis(67, "origin",           "derived",       "hybrid",        "original",      "entity"),
    Axis(68, "size_class",       "small",         "medium",        "large",         "entity"),
    Axis(69, "link_density",     "isolated",      "sparse",        "dense",         "entity"),
    Axis(70, "update_mode",      "batch",         "periodic",      "realtime",      "entity"),
]

# Layer: Field / Instance (trits 71-81)
FIELD_AXES = [
    Axis(71, "numeric_sign",     "negative",      "zero",          "positive",      "field"),
    Axis(72, "bound_type",       "lower",         "unbounded",     "upper",         "field"),
    Axis(73, "comparison",       "less_than",     "equal",         "greater_than",  "field"),
    Axis(74, "trend",            "decreasing",    "stable",        "increasing",    "field"),
    Axis(75, "anomaly",          "below_normal",  "normal",        "above_normal",  "field"),
    Axis(76, "confidence",       "low",           "medium",        "high",          "field"),
    Axis(77, "relevance",        "tangential",    "moderate",      "primary",       "field"),
    Axis(78, "recency",          "old",           "moderate",      "new",           "field"),
    Axis(79, "specificity",      "general",       "moderate",      "specific",      "field"),
    Axis(80, "affect",           "negative",      "neutral",       "positive",      "field"),
    Axis(81, "salience",         "background",    "normal",        "foreground",    "field"),
]

# Assemble all axes
ALL_AXES: List[Axis] = (
    PROTOCOL_AXES + HARDWARE_AXES + ORGANIZATION_AXES + 
    APPLICATION_AXES + ENTITY_AXES + FIELD_AXES
)

# Index by ID for quick lookup
AXIS_BY_ID: Dict[int, Axis] = {a.id: a for a in ALL_AXES}

# Group by layer
AXES_BY_LAYER: Dict[str, List[Axis]] = {}
for ax in ALL_AXES:
    AXES_BY_LAYER.setdefault(ax.layer, []).append(ax)


class SemanticDimensions:
    """Interface for working with the 81-dimensional meaning space."""
    
    def __init__(self):
        self.axes = ALL_AXES
        self.n_dims = len(self.axes)
    
    def describe(self, trits: np.ndarray) -> List[str]:
        """Human-readable description of a trit vector.
        
        Only describes non-zero dimensions (displacements from baseline).
        """
        descriptions = []
        for i, t in enumerate(trits):
            if t != 0 and i < len(self.axes):
                descriptions.append(self.axes[i].describe_trit(int(t)))
        return descriptions
    
    def layer_summary(self, trits: np.ndarray) -> Dict[str, Dict]:
        """Summarize trit vector by trust hierarchy layer."""
        summary = {}
        for layer_name, axes in AXES_BY_LAYER.items():
            active = []
            for ax in axes:
                idx = ax.id - 1  # 0-indexed
                if idx < len(trits) and trits[idx] != 0:
                    active.append(ax.describe_trit(int(trits[idx])))
            summary[layer_name] = {
                "total_dims": len(axes),
                "active_dims": len(active),
                "descriptions": active,
            }
        return summary
    
    def get_axis(self, dim: int) -> Axis:
        """Get axis definition by dimension number (1-indexed)."""
        return AXIS_BY_ID[dim]
    
    def zero_vector(self) -> np.ndarray:
        """The origin: universal baseline (all zeros)."""
        return np.zeros(self.n_dims, dtype=np.int8)
