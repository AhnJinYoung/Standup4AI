import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

MODEL_ID = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# MXFP4 → 학습 시 bf16으로 업캐스트
quant = Mxfp4Config(dequantize=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map="auto",
    quantization_config=quant,
)

# LoRA (MoE expert proj 포함)
lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ],
)
model = get_peft_model(model, lora)

# SFT 설정 (VRAM에 맞춰 조절)
train_args = SFTConfig(
    learning_rate=2e-4,
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir="gptoss-20b-comedy-sft",
    push_to_hub=True,  # 아래 push에서 모델 리포 id 지정
)

train_ds = load_dataset("parquet",data_files="prep_out/comedy_messages.parquet")["train"]
# 위 전처리에서 만든 Dataset(컬럼: "messages")
trainer = SFTTrainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    processing_class=tokenizer,     # <-- Harmony 채팅 템플릿 자동 적용
)

trainer.train()

# 학습된 LoRA 어댑터를 HF 모델 리포에 업로드
trainer.push_to_hub("YOUR_ORG_OR_USER/gptoss-20b-comedy-sft")
