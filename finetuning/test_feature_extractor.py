#!/usr/bin/env python3
"""
Gemma 3 1B IT 모델을 QLoRA 방식으로 파인튜닝하는 스크립트
데이터셋: data/feature_extraction_instruction_dataset/*.jsonl
모델: google/gemma-3-1b-it

실행 방법:
    export HF_TOKEN=<huggingface access token>
    export WANDB_API_KEY=<wandb api key>
    export HF_HOME=<huggingface cache directory>
    uv run python app/scripts/train_feature_extraction_model.py
"""

import os
from pathlib import Path
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb


def load_feature_extraction_dataset() -> Dataset:
    """feature_extraction_instruction_dataset 로드"""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_path = f"{BASE_DIR}/../data/feature_extraction_instruction_dataset"
    
    data_files = []
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        raise ValueError(f"데이터셋 경로가 존재하지 않습니다: {dataset_path}")
    
    # jsonl 파일들 찾기
    for file_path in dataset_dir.glob("*.jsonl"):
        data_files.append(str(file_path))
    
    if not data_files:
        raise ValueError(f"데이터셋 파일이 없습니다: {dataset_path}")
    
    print(f"로드할 데이터 파일: {len(data_files)}개")
    
    # 모든 jsonl 파일 로드하고 합치기
    datasets = []
    for file_path in data_files:
        dataset = load_dataset("json", data_files=file_path, split="train")
        datasets.append(dataset)
    
    # 데이터셋들을 하나로 합치기
    combined_dataset = concatenate_datasets(datasets)
    print(f"총 데이터 개수: {len(combined_dataset)}")
    
    return combined_dataset


def format_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """
    데이터셋을 모델 훈련에 적합한 형식으로 변환합니다.
    Gemma의 채팅 템플릿을 사용하여 메시지를 단일 텍스트 필드로 변환합니다.
    """
    
    def format_chat_template(example):
        messages = example["messages"]
        messages = [m for m in messages if m["role"] != "system"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # EOS 토큰이 없으면 추가
        if not text.endswith(tokenizer.eos_token):
            text = text.strip() + tokenizer.eos_token
            
        return {"text": text}

    return dataset.map(format_chat_template, remove_columns=dataset.column_names, batched=False)


def setup_model_and_tokenizer(model_id, torch_dtype, attn_implementation):
    """모델과 토크나이저 설정"""
    
    # QLoRA 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    )

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype, # QLoRA 사용 시 bnb_4bit_compute_dtype으로 지정
        device_map="auto",
        quantization_config=bnb_config,
    )
    model.config.use_cache = False

    
    # k-bit 훈련을 위한 모델 준비 (QLoRA 사용 시)
    # model = prepare_model_for_kbit_training(model)
    
    # PEFT 모델 적용
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    
    return model, tokenizer
                
def main():
    """메인 실행 함수"""
    
    # 시드 설정
    torch.manual_seed(42)

    print("=== Gemma 3 QLoRA 파인튜닝 시작 ===")

    # 1. 데이터셋 로드
    print("\n1. 데이터셋 로드 중...")
    dataset = load_feature_extraction_dataset()
    
    # 2. 모델과 토크나이저 설정
    print("\n2. 모델과 토크나이저 설정 중...")

    # Check if GPU benefits from bfloat16
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"

    print(f"torch_dtype: {torch_dtype}")
    print(f"attn_implementation: {attn_implementation}")

    model_id = "google/gemma-3-1b-it"
    
    model, tokenizer = setup_model_and_tokenizer(model_id=model_id, torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        
    # 3. 데이터 포맷팅
    print("\n3. 데이터 포맷팅 중...")
    formatted_dataset = format_dataset(dataset, tokenizer)
    
    # 4. 훈련/검증 분할
    print("\n4. 데이터 분할 중...")
    split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(eval_dataset)}개")

    print(f"{'-' * 30}데이터 샘플{'-' * 30}")
    print(f"{train_dataset[0]}")

    # 출력 디렉토리 생성
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = f"{BASE_DIR}/models/feature_extractor/{model_id}"
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(project=f"restaurant-feature-extractor-gemma-3-1b-it")
    
    # 5. 훈련 설정
    print("\n5. 훈련 설정 중...")
    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=4096,
        packing=True,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        optim="adamw_torch_fused",
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True if torch_dtype == torch.float16 else False,
        bf16=True if torch_dtype == torch.bfloat16 else False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="wandb",
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        seed=42,
        data_seed=42,
    )

    # LoRA 설정
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 6. 트레이너 설정
    print("\n6. 트레이너 설정 중...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # 7. 훈련 시작
    print("\n7. 훈련 시작...")
    trainer.train(resume_from_checkpoint=True)
    
    # 8. 모델 저장
    print("\n8. 모델 저장 중...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("\n=== 파인튜닝 완료! ===")


if __name__ == "__main__":
    main()
