# Copyright 2023 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0
# Modified from https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/merge_peft_adapter.py

import argparse
import os

import peft
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, T5ForConditionalGeneration


def parse_args():
    ''' Argument parser '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_model_path", type=str,
                        help="Path to the peft_model")
    parser.add_argument("--save_path", type=str,  help="Merged model path")

    args = parser.parse_known_args()
    return args


args, _ = parse_args()

peft_model_path = args.peft_model_path
save_path = args.save_path

print("Step 1: Loading PEFT config")
peft_config = PeftConfig.from_json_file(
    os.path.join(peft_model_path, 'adapter_config.json'))

print("Step 2: Loading Base Model")
model = T5ForConditionalGeneration.from_pretrained(
    peft_config['base_model_name_or_path'],
    return_dict=True
)
tokenizer = AutoTokenizer.from_pretrained(
    peft_config['base_model_name_or_path'])

print("Step 3: Loading LORA Model")
# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_path)
model.eval()

key_list = [key for key, _ in model.base_model.model.named_modules()
            if "lora" not in key]

print(f"Step 4: Mergining {len(key_list)} modules")

for key in key_list:
    parent, target, target_name = model.base_model._get_submodules(key)
    if isinstance(target, peft.tuners.lora.Linear):
        bias = target.bias is not None
        new_module = torch.nn.Linear(
            target.in_features, target.out_features, bias=bias)
        model.base_model._replace_module(
            parent, target_name, new_module, target)

model = model.base_model.model

print(f"Saving to {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
