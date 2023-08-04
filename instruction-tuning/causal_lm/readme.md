# Instruction Tuning LLMs

Instruction tuning Casual LLMs (LLaMA, xgen etc). <br>

Code is based on https://github.com/tatsu-lab/stanford_alpaca

### Prepare the env
Once you are in the causal_lm dir, create conda env and install requirements
```
conda env -n inst_tuning python=3.8
conda activate inst_tuning
pip install -r requirements.txt
```


### Dataset Preparation

The code works with CSV file as the training data. The CSV file needs 2 columns: `prompt` and `response`.

- `prompt` column contains the instructions being sent to the model,
- `response` column contains the responses expected from the model.

#### Prompt Template Example

Smaller models generally require some sort of prompt template. This template helps the models understand where the instruction is contained and when to stop its response.

You could use Alpaca prompt template or any other template to perform this conversion.

<b> For example:

Modify your `prompt` column such that each prompt is formatted using the below Python f-string:

```
instruction = "how do I bake a cake?"
new_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n### Response:"

print(new_prompt)
```

Output:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
how do I bake a cake?

### Response:
```

We provide you with the [VMware/Open-instruct-dolly-hhrlhf-oasst](https://huggingface.co/datasets/VMware/open-instruct-v1-oasst-dolly-hhrlhf) dataset in a CSV called `open_instruct_v1.csv`.


### Instruction tune tune Causal models


| Parameter                            | Type   | Description                                                       |
|--------------------------------------|--------|-------------------------------------------------------------------|
| nproc_per_node                       | int    | num_gpus                                                          |
| model_name_or_path                   | str    | Huggingface hub model path / local dir                             |
| data_path                            | str    | path to the csv file                                              |
| source                               | str    | source column name of the csv file                                |
| target                               | str    | target column name of the csv file                                |
| bf16                                 | bool   | Train in bf16 format or not (only works for A series GPUs, not V100s) |
| output_dir                           | str    | Dir to save the output checkpoints                                |
| num_training_epochs                  | int    | num of epochs to train the models                                 |
| per_device_train_batch_size           | int    | batch_size_per_gpu on train set                                   |
| per_device_eval_batch_size            | int    | batch_size_per_gpu on eval set (if no eval set exists, doesn't matter) |
| model_max_length                     | int    | maximum sequence length for training (Equal to source+ target tokens) |
| gradient_accumulation_steps          | int    | num of steps to withhold gradients for, before doing a single back pass. Used to synthesize larger effective batch_size for training |
| evaluation_strategy                  | bool   | set to no (currently do not support evaluation)                   |
| save_strategy                        | str    | steps (refer to hugging face trainer class params for more options) |
| save_steps                           | int    | if save_strategy is steps, num steps to save after                 |
| save_total_limit                     | int    | number of checkpoints to keep                                     |
| learning_rate                        | float  | learning rate for training the model                              |
| weight_decay                         | float  | weight decay                                                      |
| warmup_ratio                         | float  | number of training steps to use for warmup                        |
| lr_scheduler_type                    | str    | lr scheduler provided by huggingface                              |
| logging_steps                        | int    | log output after x steps                                          |
| fsdp                                 | str    | sharding strategy                                                 |
| fsdp_transformer_later_cls_to_wrap   | str    |  layer to wrap for fsdp, generally its the decoder layer |

<b> For more info on training args, refer to https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/trainer#transformers.TrainingArguments </b>

#### Example script for training llama type on open_instruct dataset

Running it on 8 A100 80GB GPUs

Total batch size = num_gpus * per_device_train_batch_size * gradient_accumulation_steps 

Total_batch_size = 8 * 2 * 8 = 128

```
torchrun --nproc_per_node=4 llama_training.py \
    --model_name_or_path openlm-research/open_llama_7b\
    --data_path ./open_instruct_v1.2.csv \
    --source prompt \
    --target response \
    --bf16 True \
    --output_dir ./open_llama_7b_open_instruct\
    --num_train_epochs 3 \
    --per_device_train_batch_size 2\
    --per_device_eval_batch_size 2 \
    --model_max_length 1024\
    --gradient_accumulation_steps 8\
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
```

#### Example script for training xgen models

Running it on 3 A100 80GB GPUs

Total batch size = num_gpus * per_device_train_batch_size * gradient_accumulation_steps 

Total_batch_size = 3 * 1 * 32 = 96

```
torchrun --nproc_per_node=3 xgen_training.py \
    --model_name_or_path Salesforce/xgen-7b-4k-base\
    --data_path ./open_instruct_v1.2.csv \
    --source prompt \
    --target response \
    --bf16 True \
    --output_dir ./xgen-7b-4k-open-instruct\
    --num_train_epochs 3 \
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1\
    --model_max_length 1024\
    --gradient_accumulation_steps 32\
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
```

#### Not enough VRAM

If you cannot fit your model into GPU and don't have enough VRAM, use deepspeed stage2 or stage 3 CPU offloading to offload computations onto CPU. This slows the model down. 
https://www.deepspeed.ai/tutorials/zero/

You can also use the scripts provided in peft-seq2seq dir to train using LoRA, pass in additional parameter --deepspeed and specifiying the ds_config.json

```
--deepspeed "ds_config.json"
```

The ds_config.json file use deepspeed stage 2