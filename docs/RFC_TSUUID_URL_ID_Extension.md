# RFC: TSUUID URL_ID Extension

**Status:** Draft
**Author:** Dennis Hay
**Date:** 2026-04-12

## Problem

TSUUIDs encode meaning in their 81 ternary trits — the UUID IS the semantic identity. But meaning alone doesn't let you retrieve the underlying resource (image, document, audio file). A separate pointer is needed.

Existing approaches use file paths or full URLs in a metadata column. These break when:
- Files move (paths become stale)
- URLs change (signed URLs expire, tokens rotate)
- The reference outlives its origin system

## Proposal: URL_ID — Immutable Short Handle Alongside the Trits

Pair every TSUUID with an optional **URL_ID**: a short, immutable, content-addressed handle to an external resource.

```
Trits (81)    = semantic identity        — stored in UUID v8
URL_ID (~8 chars) = retrieval handle    — stored alongside
```

The trits carry meaning. The URL_ID carries retrieval. Together they form a complete reference: *what it means* and *where to get it*.

## URL_ID Format

An URL_ID is a short alphanumeric handle that resolves to a canonical URL via a shortener service. Canonical forms:

```
tinyurl:2xpvlcsu       → https://tinyurl.com/2xpvlcsu
bitly:3xKJp7q          → https://bit.ly/3xKJp7q
yourls:abc123          → https://self-hosted.ly/abc123
sha256:4f3c8a...(8)    → content hash (first 8 chars)
```

The scheme prefix names the shortener; the trailing ID is the handle. A reader encountering `tinyurl:2xpvlcsu` can resolve it to the underlying resource without any other information.

## Storage

Extend the TSUUID metadata schema to include `url_id TEXT`:

```sql
CREATE TABLE tsuuid_768 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE,
    title TEXT,
    vec BLOB,
    domain TEXT,
    url_id TEXT,              -- NEW: scheme:handle
    encoded_at TEXT
);
```

## Creation Workflow (photo example)

1. Capture: `UIImage` from camera
2. Upload to cloud storage: `/TSUUID768/photos/YYYYMMDD.jpg` → shared link
3. Normalize the link (e.g., Dropbox `www.dropbox.com` → `dl.dropboxusercontent.com` for direct bytes)
4. Shorten via URL shortener → short URL + handle
5. Encode the content: CLIP(image) → Vector768, OCR(image) → Vector768
6. Project to 81 trits → pack into UUID v8
7. Store `(uuid, vec, url_id="tinyurl:2xpvlcsu")` in database

## Resolution Workflow

1. Search returns a TSUUID with `url_id="tinyurl:2xpvlcsu"`
2. Client constructs `https://tinyurl.com/2xpvlcsu`
3. HTTP GET follows the redirect to the canonical resource URL
4. Display / download as appropriate

## Why This Works

**Short.** 8 characters is plenty for 62^8 ≈ 2.18e14 unique resources per shortener.

**Immutable.** A tinyurl alias doesn't change even if the underlying Dropbox URL's signature rotates (you update the tinyurl destination, the ID stays).

**Portable.** Any client with HTTP can resolve it. No SDK required. No auth required for public resources.

**Composable.** Multiple TSUUIDs can share the same URL_ID (e.g., text + image encoding of the same source). Reverse lookup is also trivial — all vectors with the same handle describe the same resource.

**Orthogonal to the trits.** The URL_ID is retrieval metadata. Two documents with identical meaning (same trits) can live at different URL_IDs; two different documents can point at the same URL_ID via deduplication.

## Non-goals

- **Not a replacement for content-addressed storage.** URL_IDs point at a mutable target. For immutable content addressing, use `sha256:` prefix.
- **Not encryption.** The URL_ID is a public handle.
- **Not embedded in the UUID.** The trits are fully committed to meaning. URL_ID is a parallel field.

## Reference Implementation

See `TSUUID768/TSUUID768/Services/PhotoUploadService.swift` for a Swift implementation using Dropbox + tinyurl's free `api-create.php` endpoint.

## Acknowledgement

This extension was proposed by Dennis Hay during iPhone app development, 2026-04-12. The original insight: "the 8-char tinyurl id could simply be used to link back to the dropbox or any other link."
