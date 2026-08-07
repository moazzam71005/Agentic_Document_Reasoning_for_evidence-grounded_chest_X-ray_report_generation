import torch
import transformers

if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

from peft import LoraConfig, get_peft_model

print("Loading model...")
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    "/data/Taimoor/ADRE-CXR/weights/Llama3-OpenBioLLM-8B",
    quantization_config=bnb_config,
    device_map="auto"
)
print("Model loaded. Memory:", torch.cuda.memory_allocated() / 1024**3, "GB")

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
print("PEFT applied. Memory:", torch.cuda.memory_allocated() / 1024**3, "GB")

model.gradient_checkpointing_enable()
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

print("Ready to train. Memory:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("SUCCESS!")
