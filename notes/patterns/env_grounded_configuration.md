# Environment Grounded Configuration

This note documents API key and configuration management.

## Coding Convention
Sensitive information, API credentials, and default execution backbones must be managed via `.env` files and validated programmatically rather than hardcoded.

## Details
* **dotenv**: Load environment variables on startup using `dotenv.load_dotenv()`.
* **Config Verification**: Core classes validate their required environment parameters (e.g. `GROQ_API_KEY`, `GEMINI_API_KEY`) at initialization time, raising explanatory `ValueError`s if keys are missing.
* **Fallback Values**: Use default environment variables for default models to simplify testing across environments.

## Usage Example
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Client:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
```

---
*Related: [[patterns-moc]]*
