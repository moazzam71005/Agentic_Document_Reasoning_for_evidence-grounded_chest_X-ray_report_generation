# Llama3-OpenBioLLM-8B Model Loading and Tokenizer Issues

Link back to: [[notes/context/overview.md|Project Overview]]

## 1. PyTorch version restriction check
### Problem Description
When attempting to load local model weights stored in `.bin` files (instead of `.safetensors`), `transformers` raised a `ValueError` under PyTorch 2.5.1, stating that PyTorch 2.6+ is required to load models safely.

### Solution
Monkeypatched `check_torch_load_is_safe` in `transformers.utils.import_utils` and `transformers.modeling_utils` to bypass this check:
```python
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
```

## 2. Missing chat template in tokenizer
### Problem Description
The `tokenizer_config.json` for Llama-3-OpenBioLLM-8B does not define a `"chat_template"` field, causing `tokenizer.apply_chat_template` to fail with a `ValueError`.

### Solution
Programmatically set the tokenizer's chat template to the standard Llama 3 instruct template when loading the model:
```python
if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
        "{% endif %}"
    )
```
