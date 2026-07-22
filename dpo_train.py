#!/usr/bin/env python3
"""
DPO Training for google/translategemma-4b-it
Dataset: Helsinki-NLP/opus-100 en-gu
Task: teach model to use । instead of . at sentence endings

Single GPU test:
    CUDA_VISIBLE_DEVICES=1 python dpo_train.py

Both GPUs:
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
        --config_file accelerate_config.yaml dpo_train.py
"""

import json
import logging
import argparse
from pathlib import Path

import torch
import sacrebleu
from datasets import load_dataset
from evaluation.prompts import build_prompt
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer
from accelerate import PartialState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PurnaViramEvalCallback(TrainerCallback):
    """
    Fires every eval_every_n_steps.
    Runs inference on num_test_samples from the test set.
    Logs purna_viram_accuracy and BLEU to logs/inference_results.jsonl
    """

    def __init__(
        self,
        tokenizer,
        test_dataset,
        num_test_samples=50,
        max_new_tokens=128,
        log_file="logs/inference_results.jsonl",
    ):
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.num_test_samples = min(num_test_samples, len(test_dataset))
        self.test_subset = test_dataset.select(range(self.num_test_samples))
        self.max_new_tokens = max_new_tokens
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        args,
        state,
        control,
        model=None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        logger.info("=" * 80)
        logger.info("Running validation after epoch %.2f", state.epoch)
        logger.info("=" * 80)

        model.eval()

        hits = 0
        hypotheses = []
        references = []
        sample_results = []

        with torch.no_grad():
            for example in self.test_subset:

                prompt = example["prompt"]
                chosen = example["chosen"]

                tokenizer_kwargs = {
                    "return_tensors": "pt",
                    "truncation": True,
                    "max_length": 256,
                }

                if "gemma" in model.config.model_type.lower():
                    tokenizer_kwargs["add_special_tokens"] = False

                inputs = self.tokenizer(
                    prompt,
                    **tokenizer_kwargs,
                ).to(model.device)

                if "gemma" not in model.config.model_type.lower():
                    inputs.pop("token_type_ids", None)

                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                generated = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                ends_purna = generated.endswith("।")
                ends_dot = generated.endswith(".")

                if ends_purna:
                    hits += 1

                hypotheses.append(generated)
                references.append(chosen)

                sample_results.append({
                    "prompt": prompt,
                    "generated": generated,
                    "chosen": chosen,
                    "ends_।": ends_purna,
                    "ends_.": ends_dot,
                })

        accuracy = hits / self.num_test_samples
        bleu = sacrebleu.corpus_bleu(
            hypotheses,
            [references],
        )

        logger.info(
            "Epoch %.2f | Accuracy %.2f%% | BLEU %.2f",
            state.epoch,
            accuracy * 100,
            bleu.score,
        )

        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 2),
            "accuracy": accuracy,
            "bleu": bleu.score,
            "samples": sample_results[:5],
        }

        with open(
            self.log_file,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        model.train()

def load_model_and_tokenizer(model_name, load_in_4bit=False):
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        add_eos_token=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    logger.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if load_in_4bit else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    return model, tokenizer


def build_lora_config(r, alpha, dropout):
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_dpo_dataset(path):
    ds = load_dataset("json", data_files=path, split="train")
    for col in ("prompt", "chosen", "rejected"):
        if col not in ds.column_names:
            raise ValueError(f"Missing column '{col}' in {path}")
    extra = [c for c in ds.column_names
             if c not in ("prompt", "chosen", "rejected")]
    if extra:
        ds = ds.remove_columns(extra)
    logger.info("Loaded %d rows from %s", len(ds), path)
    return ds
def main():
    state = PartialState()

    logger.info("Process index: %d", state.process_index)
    logger.info("Local process index: %d", state.local_process_index)
    logger.info("Device: %s", state.device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",         default="google/translategemma-4b-it")
    parser.add_argument("--load_in_4bit",        action="store_true", default=False)
    parser.add_argument("--train_data",          default="./data/train_dpo.jsonl")
    parser.add_argument("--test_data",           default="./data/val_dpo.jsonl")
    parser.add_argument("--output_dir",          default="./outputs/translategemma-dpo")
    parser.add_argument("--num_epochs",          type=int,   default=3)
    parser.add_argument("--batch_size",          type=int,   default=1)
    parser.add_argument("--grad_accum",          type=int,   default=8)
    parser.add_argument("--learning_rate",       type=float, default=5e-5)
    parser.add_argument("--max_seq_length",      type=int,   default=512)
    parser.add_argument("--max_prompt_len",      type=int,   default=256)
    parser.add_argument("--lora_r",              type=int,   default=64)
    parser.add_argument("--lora_alpha",          type=int,   default=128)
    parser.add_argument("--lora_dropout",        type=float, default=0.05)
    parser.add_argument("--beta",                type=float, default=0.1)
    parser.add_argument("--num_test_samples",    type=int,   default=50)
    parser.add_argument("--logging_steps",       type=int,   default=10)
    parser.add_argument("--report_to",           default="none")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("DPO Training — TranslateGemma 4B — Gujarati DPO")
    logger.info("Model:  %s", args.model_name)
    logger.info("Train:  %s", args.train_data)
    logger.info("Test:   %s", args.test_data)
    logger.info("GPUs:   %d", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory // (1024**3)
        logger.info("  GPU %d: %s (%dGB)", i,
                    torch.cuda.get_device_name(i), mem)
    logger.info("=" * 60)

    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.load_in_4bit
    )
    lora_config   = build_lora_config(
        args.lora_r, args.lora_alpha, args.lora_dropout
    )
    train_dataset = load_dpo_dataset(args.train_data)
    test_dataset  = load_dpo_dataset(args.test_data)

    def preprocess(example):
        example["prompt"] = build_prompt(
            example,
            model_family="translategemma",
            tokenizer=tokenizer,
        )
        return example

    train_dataset = train_dataset.map(preprocess)
    print("=" * 80)
    print(train_dataset[0]["prompt"])
    print("=" * 80)
    test_dataset = test_dataset.map(preprocess)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="no",
        report_to=args.report_to,
        remove_unused_columns=False,
        beta=args.beta,
        max_length=args.max_seq_length,
        seed=42,
    )

    # callback = PurnaViramEvalCallback(
    #     tokenizer=tokenizer,
    #     test_dataset=test_dataset,
    #     num_test_samples=args.num_test_samples,
    #     log_file=Path(args.output_dir) / "logs" / "inference_results.jsonl",
    # )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=lora_config,
        # callbacks=[callback],
        callbacks=[]
    )

    trainable = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in trainer.model.parameters())
    logger.info(
        "Trainable: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    logger.info("Starting DPO training...")
    result = trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    logger.info("Done. Adapter saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
