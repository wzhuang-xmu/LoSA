import torch
import torch.nn as nn
import gc
import json
import time
from CKA import cka
import numpy as np
from peft.tuners.lora import LoraLayer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    )
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model
    )
from datasets import load_dataset, Dataset, DatasetDict
from trl import SFTTrainer
from lib.eval import eval_ppl
from lib.prune import check_sparsity, prune_sparsegpt, prune_wanda, get_feature_map, allocate_ranks
from lib.data import get_loaders
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='LLaMA model')
parser.add_argument('--seed', type=int, default=4, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
parser.add_argument("--prune_method", type=str, choices=["wanda", "sparsegpt"])
parser.add_argument("--cache_dir", default="llm_weights", type=str )
parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
parser.add_argument('--save', type=str, default=None, help='Path to save results.')
parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
args = parser.parse_args()

print("loading model...")
print(args.model)
start_time = time.time()

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)

model_seqlen = 2048
model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()
model.config.use_cache = False
model.config.pretraining_tp = 1

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    use_fast=False,
    padding_side='right',
    trust_remote_code=True,
    add_eos_token=True,
    add_bos_token=True
    )

tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj',
        ],
    bias="none",
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
    )

model = get_peft_model(model, peft_config)

c4_dataloader, _ = get_loaders(
    "c4",
    nsamples=args.nsamples,
    seed=args.seed,
    seqlen=model_seqlen,
    tokenizer=tokenizer,
)

if '7' in args.model or '8' in args.model:
    calib_nsamples=64
elif '13' in args.model:
    calib_nsamples=32
calib_dataloader, _ = get_loaders(
    "c4",
    nsamples=calib_nsamples,
    seed=args.seed,
    seqlen=model_seqlen,
    tokenizer=tokenizer,
)

# Handling n:m sparsity
prune_n, prune_m = 0, 0
model.base_model.model.seqlen = model_seqlen

model.base_model.model.eval()

model.merge_adapter()
dense_feature = get_feature_map(args, model.base_model.model, tokenizer, dataloader=c4_dataloader)
model.unmerge_adapter()

dense_feature = [feature.cpu() for feature in dense_feature]

torch.cuda.empty_cache()
gc.collect()

iters = 5
iter_mean_rank = 6
split_size = 1000

for prune_iter in range(1, iters+1):
    model.merge_adapter()

    if args.prune_method=='wanda':
        print('prune_wanda')
        recon_loss = prune_wanda(args, model.base_model.model, tokenizer, prune_n=prune_n, dense_feature = dense_feature,
                                prune_m=prune_m, dataloader=c4_dataloader, calib_dataloader = calib_dataloader,
                                prune_iter=prune_iter, iters=iters)
    else:
        print('prune_sparsegpt')
        recon_loss = prune_sparsegpt(args, model.base_model.model, tokenizer, prune_n=prune_n, dense_feature = dense_feature,
                                prune_m=prune_m, dataloader=c4_dataloader, calib_dataloader = calib_dataloader,
                                prune_iter=prune_iter, iters=iters)

    model.unmerge_adapter()

    iter_rank = allocate_ranks(np.array(recon_loss), rank=iter_mean_rank)
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            setattr(module, 'iter_rank', iter_rank[layer_idx])
        if 'post_attention_layernorm' in name:
            layer_idx += 1

    iter_mean_rank += 1

    if '7' in args.model or '8' in args.model:
        train_batch_size=4
    elif '13' in args.model:
        train_batch_size=2

    training_arguments = TrainingArguments(
        output_dir= './results',
        num_train_epochs= 1,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=2,
        optim = 'paged_adamw_8bit',
        save_steps= 1000,
        logging_steps= 20,
        learning_rate= 2e-4,
        weight_decay= 0.001,
        fp16=False,
        bf16=False,
        max_grad_norm= 0.3,
        max_steps= -1,
        warmup_ratio= 0.3,
        group_by_length= True,
        lr_scheduler_type= 'linear',
        report_to="none"
        )

    original_dataset = 'vicgalle/alpaca-gpt4'
    start_index = prune_iter * split_size
    end_index = start_index + split_size
    split = f"train[{start_index}:{end_index}]"
    if prune_iter==5:
        split = f"train[4000:10000]"
    dataset = load_dataset(original_dataset, split=split)

    # Set Supervised Finetuning Trainer (SFTTrainer) parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=1024,
        dataset_text_field='text',
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False
        )

    # Train model
    trainer.train()

    model.merge_adapter()
    ################################################################
    print("*"*30)
    print("After train sparsity checking...")
    sparsity_ratio = check_sparsity(model.base_model.model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    model.unmerge_adapter()

end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"All running time: {elapsed_time:.2f} min")


model.merge_and_unload()

device = torch.device("cuda:0")
dataset = 'wikitext2'
model.base_model.model.seqlen = 4096
ppl = eval_ppl(model, tokenizer, dataset, device)
print(f"\nppl on {dataset}: {ppl}\n")