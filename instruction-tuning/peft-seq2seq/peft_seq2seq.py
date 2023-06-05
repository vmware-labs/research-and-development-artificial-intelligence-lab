# Copyright 2023 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import logging
import os
import threading

import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from deepspeed.accelerator import get_accelerator
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          get_linear_schedule_with_warmup, set_seed)


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
                        default=2, help="Batch size to use for training.")
    parser.add_argument("--input_max_length", type=int, default=128,
                        help="Maximum input length to use for generation")
    parser.add_argument("--target_max_length", type=int, default=128,
                        help="Maximum target length to use for generation")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed to use for training.")
    parser.add_argument("--input_column", type=str,
                        default='input', help='csv input text column')
    parser.add_argument("--target_column", type=str,
                        default='output', help='csv target text column')
    parser.add_argument("--save_path", type=str,
                        default='peft_ckpt', help="Save path")

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

    # Create  dir if it doesn't exist
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
    accelerator = Accelerator()
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8,
        lora_alpha=32, lora_dropout=0.1
    )
    set_seed(seed)

    # Save logs only on the main process
    @accelerator.on_main_process
    def log_info(logging, s):
        logging.info(s)

    # load dataset
    dataset = load_dataset('csv', data_files={'train': data_file})
    log_info(logging, f"Dataset length :{len(dataset['train'])}")
    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    # load peft model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def preprocess_function(sample, padding="max_length"):
        # created prompted input
        inputs = sample[text_column]
        # tokenize inputs
        model_inputs = tokenizer(
            inputs, max_length=source_max_length,
            padding=padding, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample[label_column],
                           max_length=target_max_length,
                           padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id
        # in the labels by -100 when we want to ignore padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Prepare and preprocess the dataset
    with accelerator.main_process_first():
        # preventing string conversion errors

        def str_convert(example):
            example[label_column] = str(example[label_column])
            return example

        dataset['train'] = dataset['train'].map(str_convert)
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()
    train_dataset = processed_datasets["train"]

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=batch_size, pin_memory=True
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # accelerator prepapre
    model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, optimizer, lr_scheduler
    )

    # Train the model
    for epoch in range(num_epochs):
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                # using accelerator accumulate to perform gradient accumulation
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    get_accelerator().empty_cache()

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
        log_info(logging, f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        # save intermediate checkpoint
        log_info(logging, "Saving intermediate ckpt")
        accelerator.wait_for_everyone()
        success = model.save_checkpoint(f'{save_path}', f'{epoch}')
        # save peft config
        peft_config.save_pretrained(os.path.join(f'{save_path}', f'{epoch}'))

        status_msg = f"checkpointing: checkpoint_folder={save_path}"
        if success:
            log_info(logging, f"Success {status_msg}")
        else:
            log_info(logging, f"Failure {status_msg}")

    log_info(logging, "Training complete ......")


if __name__ == "__main__":
    main()
