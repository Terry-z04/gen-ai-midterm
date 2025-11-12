
# """
# Supervised fine-tuning (SFT) with QLoRA on a small instruction dataset.

# Usage:
#   accelerate launch finetuning/finetune_sft.py \
#     --base_model Qwen/Qwen2.5-3B-Instruct \
#     --train_path finetuning/train.jsonl \
#     --val_path finetuning/val.jsonl \
#     --output_dir qlora-ads \
#     --batch_size 4 --grad_accum 4 --epochs 3 --lr 2e-5
# """
# import os, json, math, argparse
# from dataclasses import dataclass
# from typing import Dict, List
# from datasets import load_dataset
# import torch
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
#     TrainingArguments, Trainer
# )
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from transformers import BitsAndBytesConfig

# PROMPT_TMPL = """You are a helpful admissions assistant for UChicago's MS in Applied Data Science.
# Use clear, current program language where possible and avoid making up facts.

# Instruction: {instruction}
# User Question: {input}
# Helpful Answer:"""

# def format_example(ex):
#     return PROMPT_TMPL.format(instruction=ex["instruction"], input=ex["input"]) + " " + ex["output"]

# def get_args():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
#     ap.add_argument("--train_path", type=str, default="finetuning/train.jsonl")
#     ap.add_argument("--val_path", type=str, default="finetuning/val.jsonl")
#     ap.add_argument("--output_dir", type=str, default="qlora-ads")
#     ap.add_argument("--batch_size", type=int, default=4)
#     ap.add_argument("--grad_accum", type=int, default=4)
#     ap.add_argument("--epochs", type=int, default=3)
#     ap.add_argument("--lr", type=float, default=2e-5)
#     ap.add_argument("--max_len", type=int, default=1024)
#     return ap.parse_args()

# def main():
#     args = get_args()

#     # Tokenizer & model
#     tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # 4-bit quantization for memory efficiency
#     bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
#     model = AutoModelForCausalLM.from_pretrained(
#         args.base_model,
#         quantization_config=bnb_config,
#         device_map="auto"
#     )
#     model = prepare_model_for_kbit_training(model)

#     lora = LoraConfig(
#         r=16, lora_alpha=32, lora_dropout=0.05,
#         target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
#     )
#     model = get_peft_model(model, lora)

#     # Data
#     ds = load_dataset("json", data_files={"train": args.train_path, "validation": args.val_path})
#     def tok(batch):
#         texts = [format_example(ex) for ex in batch["instruction"]]
#         # The above would be wrong; we need the whole example dict. Let's rebuild using zip.
#         return {}

#     # Proper tokenization
#     def format_many(batch):
#         outs = []
#         for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
#             text = PROMPT_TMPL.format(instruction=inst, input=inp) + " " + out
#             outs.append(text)
#         toks = tokenizer(outs, truncation=True, max_length=args.max_len, padding="max_length")
#         toks["labels"] = toks["input_ids"].copy()
#         return toks

#     ds = ds.map(format_many, batched=True, remove_columns=ds["train"].column_names)

#     collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     # Training
#     steps_per_epoch = math.ceil(len(ds["train"]) / (args.batch_size * args.grad_accum))
#     warmup_steps = max(10, int(0.03 * steps_per_epoch * args.epochs))
#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         num_train_epochs=args.epochs,
#         learning_rate=args.lr,
#         logging_steps=5,
#         evaluation_strategy="steps",
#         eval_steps=steps_per_epoch,  # once per epoch
#         save_steps=steps_per_epoch,
#         save_total_limit=2,
#         bf16=torch.cuda.is_available(),
#         fp16=not torch.cuda.is_available(),
#         report_to="none",
#         ddp_find_unused_parameters=False
#     )

#     trainer = Trainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         data_collator=collator,
#         train_dataset=ds["train"],
#         eval_dataset=ds["validation"]
#     )

#     trainer.train()
#     trainer.save_model(args.output_dir)
#     tokenizer.save_pretrained(args.output_dir)

#     print("Done. Adapter and tokenizer saved to:", args.output_dir)

# if __name__ == "__main__":
#     main()
