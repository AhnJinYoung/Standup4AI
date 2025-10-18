from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto"
)
merged = PeftModel.from_pretrained(base, "YOUR_ORG_OR_USER/gptoss-20b-comedy-sft").merge_and_unload()

messages = [{"role": "user", "content": "Do a short stand-up bit about procrastination, keep stage directions."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(merged.device)
ids = merged.generate(**inputs, max_new_tokens=300, temperature=0.9)
print(tokenizer.decode(ids[0]))
