# Copyright 2023 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import logging
import os
import threading
from typing import Dict, Optional, Sequence
import copy
import logging
import pandas as pd
from dataclasses import dataclass, field
import psutil
import torch
import matplotlib.pyplot as plt
from accelerate import Accelerator
from transformers.tokenization_utils import AddedToken
# from deepspeed.accelerator import get_accelerator
from peft import LoraConfig, TaskType, get_peft_model
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

import transformers

def plot_loss(steps, batch_loss, total_loss, filepath):
    """
    Plots the loss against the steps and saves the plot as an image file.

    Args:
    - steps: a list of integers representing the steps
    - loss: a list of floats representing the loss
    - filepath: a string representing the file path to save the plot image
    """
    plt.plot(steps, batch_loss, label = 'step_loss')
    plt.plot(steps, total_loss, label ='total_loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(filepath,'loss_plot.png'))
    plt.close()


def b2mb(x):
    '''
    Converting Bytes to Megabytes
    '''
    return int(x / 2**20)


class TorchTracemalloc:
    '''
    # Context manager is used to track the peak memory usage of the process
    '''

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1
        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False
        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)
        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
IGNORE_INDEX = -100

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
#Save plot


# Handle argument parsing
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help="Model path. Supports T5/UL2 models")
    parser.add_argument("--datafile_path", type=str, default="sample.csv",
                        help="Path to the already processed dataset.")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of epochs to train for.")
    parser.add_argument("--per_device_batch_size", type=int,
                        default=2, help="Batch size per model to use for training.")
    parser.add_argument("--input_max_length", type=int, default=1024,
                        help="Maximum input length to use for generation")
    parser.add_argument("--target_max_length", type=int, default=1024,
                        help="Maximum target length to use for generation")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed to use for training.")
    parser.add_argument("--input_column", type=str,
                        default='input', help='csv input text column')
    parser.add_argument("--target_column", type=str,
                        default='output', help='csv target text column')
    parser.add_argument("--save_path", type=str,
                        default='peft_ckpt', help="Save path")
    parser.add_argument("--gradient_acc", type=int,
                        default=20, help="gradient acc")

    args = parser.parse_known_args()
    return args


# Main function
def main():
    args, _ = parse_args()
    text_column = args.input_column
    label_column = args.target_column
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.per_device_batch_size
    seed = args.seed
    model_name_or_path = args.model_path
    data_file = args.datafile_path
    save_path = args.save_path
    target_max_length = args.target_max_length
    source_max_length = args.input_max_length
    gacc = args.gradient_acc

    # Create dir if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_path, "training_log.log")),
            logging.StreamHandler()
        ]
    )

    logging.info(f'Args:\n {args}')

    # launch configs
    accelerator = Accelerator(gradient_accumulation_steps=gacc)


    set_seed(seed)

    # Save logs only on the main process
    @accelerator.on_main_process
    def log_info(logging, s):
        logging.info(s)

    
    
   
    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=target_max_length,
        trust_remote_code=True,
    )


    # Add new line token
#     tokenizer.add_tokens([AddedToken("\n", normalized=False),AddedToken('`', normalized = False)])

     # load model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # reize embeddings with \n 
#     model.resize_token_embeddings(len(tokenizer))

    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] ='<|endoftext|>'
    
    pad_token = AddedToken('<|endoftext|>', lstrip=False, rstrip=False)
    tokenizer.pad_token = pad_token
    tokenizer.eos_token = pad_token

    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=special_tokens_dict,
    #     tokenizer=tokenizer,
    #     model=model,
    # )

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
        "q_proj",
        "v_proj",
    ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

#
    # load peft model
    model = get_peft_model(model, config)
    model.print_trainable_parameters()




    def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=target_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )


    def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)


    class SupervisedDataset(Dataset):
        """Dataset for supervised fine-tuning."""

        def __init__(self,  tokenizer: transformers.PreTrainedTokenizer):
            super(SupervisedDataset, self).__init__()
            # load dataset
            dataset = pd.read_csv(data_file)
            # load_dataset('csv', data_files={'train': data_file})
            log_info(logging, f"Dataset length :{len(dataset)}")

            dataset[label_column] = [f'{x}{tokenizer.eos_token}' for x in dataset[label_column]]

            data_dict = preprocess(dataset[text_column], dataset[label_column], tokenizer)

            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])


    @dataclass
    class DataCollatorForSupervisedDataset(object):
        """Collate examples for supervised fine-tuning."""


        def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
            )


    def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Make dataset and collator for supervised fine-tuning."""
        train_dataset = SupervisedDataset(tokenizer=tokenizer)
        data_collator = DataCollatorForSupervisedDataset()
        return train_dataset, data_collator



    # Prepare and preprocess the dataset
    with accelerator.main_process_first():
        # preventing string conversion errors
        train_dataset, collate_fn = make_supervised_data_module(tokenizer=tokenizer)


    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=batch_size, pin_memory=True
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.03 * (len(train_dataloader) * num_epochs),
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # lr scheduler
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=10,
    #     num_training_steps=(len(train_dataloader) * num_epochs),
    # )

    # accelerator prepapre
    model, train_dataloader, optimizer,lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )

    loss_list = []
    steps_list = []
    total_loss_list = []
    total_loss_list
    step_count = 0
    total_loss_epochs = 0
    # Train the model
    for epoch in range(num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(pbar := tqdm(train_dataloader)):
                # using accelerator accumulate to perform gradient accumulation
                with accelerator.accumulate(model):
                    
                    outputs = model(**batch)
                    loss = outputs.loss
                    current_loss = loss.detach().float()
                    total_loss += loss.detach().float()
                    total_loss_epochs+= loss.detach().float()
                    pbar.set_description(f"step loss: {current_loss}")
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    if accelerator.is_main_process:
                        step_count+=1
                        steps_list.append(step_count)
                        loss_list.append(current_loss.cpu())
                        total_loss_list.append(total_loss_epochs.cpu()/step_count)
                        if step_count%500 == 0:
                            plot_loss(steps_list,loss_list,total_loss_list,save_path)



        # Printing the GPU memory usage details
        log_info(logging,
                 "GPU Peak Memory consumed during train: {}".format(tracemalloc.peaked))
        log_info(logging,
                 "GPU Total Peak Memory consumed during the train: {}".format(
                     tracemalloc.peaked + b2mb(tracemalloc.begin)
                 )
                 )
        log_info(
            logging, "CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        log_info(logging,
                 "CPU Total Peak Memory consumed during the train (max): {}".format(
                     tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
                 )
                 )

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        log_info(
            logging, "........................ : TRAINING DETAILS : .......................")
        log_info(logging, f"{epoch}: {train_ppl} {train_epoch_loss}")

        # save intermediate checkpoint
        log_info(logging, "Saving intermediate ckpt")
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
                save_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
        # success = model.save_checkpoint(f'{save_path}', f'{epoch}')
        # # save peft config
        # # config.save_pretrained(os.path.join(f'{save_path}', f'{epoch}'))

        # status_msg = f"checkpointing: checkpoint_folder={save_path}"
        # if success:
        #     log_info(logging, f"Success {status_msg}")
        # else:
        #     log_info(logging, f"Failure {status_msg}")

    log_info(logging, "Training complete ......")


if __name__ == "__main__":
    main()
