# Modular Directory Layout

This note documents the division between pipeline source code and run scripts.

## Coding Convention
Code logic must be separated into a clean source folder (`src/`) and a command-line script folder (`scripts/`).

## Details
* **src/**: Houses core reusable models, utilities, classes, and logic (e.g. `src/llm/`, `src/agent/`, `src/evidence/`). Files here do not contain script invocation blocks (`if __name__ == "__main__":`).
* **scripts/**: Contains runnable wrapper scripts that parse command-line arguments using `argparse`, orchestrate execution, and print outputs.
* **configs/**: Stores YAML-based parameter configurations.

## Usage Example
A wrapper script `scripts/run_pipeline.py` imports functions or classes from `src/agent/controller.py` and coordinates execution:
```python
from src.agent.controller import AgentController

def main():
    # argument parsing ...
    controller = AgentController(...)
    controller.run(...)
```

---
*Related: [[patterns-moc]]*
