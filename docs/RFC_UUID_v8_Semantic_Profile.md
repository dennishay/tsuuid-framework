# UUID v8 Semantic Profile — DRAFT

**Status:** Preliminary Draft (not yet submitted to IETF)  
**Author:** Dennis Evan Hay  
**Date:** March 2026  
**Extends:** RFC 9562 §5.8 (UUID Version 8)

## Abstract

This document defines a profile of UUID Version 8 (UUIDv8) as specified in RFC 9562 for encoding semantic meaning as 81-dimensional ternary displacements. The profile uses the 122 custom bits available in UUIDv8 to encode ternary values {-1, 0, +1} representing independent axes of semantic association.

## 1. Introduction

RFC 9562 defines UUIDv8 as a format for "experimental or vendor-specific use cases" with 122 bits available for implementation-specific content. This profile assigns semantic structure to those bits.

## 2. Bit Layout

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     trit_groups_a (48 bits)                    |
|                  ~30 trits: protocol + hardware                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| ver=0b1000  |  trit_groups_b (12 bits)  ~7 trits: org/app     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|var=0b10|           trit_groups_c (62 bits)                     |
|        |      ~39 trits: entity + field dimensions             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    trit_groups_c continued                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Total custom bits: 48 + 12 + 62 = 122 bits
Ternary capacity: ⌊122 / 1.58⌋ = 77 trits (minimum)
With efficient packing (groups of 5): up to 81 trits
```

## 3. Trit Packing

Groups of 5 trits are packed into 8 bits using base-3 encoding:

```
value = t₀ + 3·t₁ + 9·t₂ + 27·t₃ + 81·t₄

where each tᵢ ∈ {0, 1, 2} (mapped from {-1, 0, +1})
range: [0, 242] (fits in uint8, since 3⁵ = 243)
```

## 4. Dimensional Axis Registry

Each trit position corresponds to a registered semantic axis. The axis registry is maintained in this repository at `src/tsuuid/dimensions.py`.

Axes are organized by trust hierarchy layer:

| Trit Range | Layer | Authority |
|---|---|---|
| 1–10 | Protocol / Standards | Standards body |
| 11–20 | Hardware / Platform | Silicon vendor |
| 21–35 | Organization / Enterprise | Enterprise IT |
| 36–55 | Application / Domain | Application owner |
| 56–70 | Entity / Record | Data steward |
| 71–81 | Field / Instance | Record originator |

## 5. Conformance Requirements

A conforming implementation MUST:
- Set UUID version bits to 0b1000 (version 8)
- Set UUID variant bits to 0b10 (RFC 9562)
- Use the trit packing algorithm defined in §3
- Map trits to registered dimensional axes

A conforming implementation SHOULD:
- Only modify trits within its authorized trust layer
- Generate deterministic UUIDs for identical semantic content
- Support the full 81-dimensional axis set

## 6. Security Considerations

The semantic UUID does not provide confidentiality. The content encoded in the ternary dimensions is readable by any system with the dimensional axis registry. Applications requiring confidentiality MUST apply additional encryption.

Semantic validation is inherent: a random or malicious trit pattern applied to a universal reference model produces incoherent output, providing a natural integrity check.

## 7. IANA Considerations

This document requests no IANA actions at this time. Future versions may request registration of a UUID v8 subtype for semantic UUIDs.

## 8. References

- RFC 9562: Davis, K., Peabody, B., Leach, P. "Universally Unique IDentifiers (UUIDs)." IETF, May 2024.
- Ma, S. et al. "The Era of 1-bit LLMs." arXiv:2402.17764, 2024.
- Hay, D. E. "Ternary Semantic UUID: A Universal Information Framework." 2026.
