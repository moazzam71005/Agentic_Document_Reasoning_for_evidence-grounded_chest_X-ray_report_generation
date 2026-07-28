# JSON Schema Validation

This note details the parsing and repair conventions for structured LLM outputs.

## Coding Convention
All structured JSON responses from LLM clients must be parsed defensively to tolerate markdown code block fences and validated before downstream consumption.

## Details
* **Fenced JSON Parsing**: A centralized parser (`parse_json_response`) strips leading/trailing markdown fences (e.g. ` ```json ` and ` ``` `) from the raw completion before calling `json.loads`.
* **Validation & Repair Loop**: If parsing or schema validation fails, a repair script (`repair.py`) is invoked, prompting the LLM with the invalid content and structural requirements to self-correct.

## Usage Example
```python
def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)
```

---
*Related: [[patterns-moc]]*
