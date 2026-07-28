# Type Annotations

This note documents the project's static typing convention.

## Coding Convention
All Python code files must utilize strict static type annotations. The standard import `from __future__ import annotations` is required at the top of each script to enable postponed evaluation of type annotations.

## Details
* Function signatures must specify parameter types and return types (e.g. `def run(...) -> EvidenceState`).
* Complex types should use generic collections (like `list[dict[str, Any]]` or `str | None`) rather than broad fallback types.
* This convention guarantees compatibility, prevents type coercion bugs, and improves IDE autocompletion.

## Usage Example
```python
from __future__ import annotations
from typing import Any

def process_data(data: list[dict[str, Any]]) -> str | None:
    # Logic here
    return None
```

---
*Related: [[patterns-moc]]*
