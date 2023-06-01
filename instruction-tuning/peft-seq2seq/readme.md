# PEFT Flan-ul2 Finetuning

Finetune Flan, T5-XXL and other larger seq2seq models using  PEFT and Deepspeed's cpu offloading on smaller GPUs. <br>

(Code is based on https://github.com/huggingface/peft/blob/main/examples/conditional_generation/peft_lora_seq2seq_accelerate_ds_zero3_offload.py)


Create the env to run using (tested on machines running cuda 11 machines):
```
conda env create --file=conda_env.yml
conda activate lora_training
```

Sample dataset is a sample of Bigscience-P3's dream_answer_to_dialogue subset where the task is given a prompt create a dialogue


Generate the accelerate and deepspeed config file (refer to the template_config.yaml file for reference)
```                     
accelerate config --config_file launcher_config.yaml
```
IMPORTANT: Train using either bf16 or fp32, t5 models overflow with fp16.

Run the script using : 
```
accelerate launch --config_file accelerate_config.yaml peft_seq2seq.py \
    --model_path google/flan-t5-large \
    --datafile_path sample.csv \
    --num_epochs 1 \
    --per_device_batch_size 8\
    --input_max_length 128\
    --target_max_length 128\
    --lr 5e-4\
    --save_path sample_ul2_ckpt\
    --gradient_accumulation_steps 8
```
<hr>

## To train the models (Flan-UL2, T5 etc ..) on the Alpaca dataset, download the dataset: 

```
wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
```

Convert the alpaca_data.json into csv format using

```
python convert_alpaca_dataset.py
```

Modify the launcher_config.yaml to tune the gradient_accumulation_steps

```
accelerate launch --config_file dummy_config.yaml peft_seq2seq.py \
    --model_path google/flan-t5-xl \
    --datafile_path alpaca_data.csv \
    --num_epochs 1 \
    --per_device_batch_size 8\
    --input_max_length 256\
    --target_max_length 128\
    --lr 1e-4\
    --save_path alpaca_ul2_ckpt
```

Convert the deepspeed checkpoints to a pytorch_model file using:

```
python ./checkpointfolder/zero_to_fp32.py ./checkpointfolder/ ./checkpointfolder/ckpt_number/adapter_model.bin
```
example:

python ./sample_ul2_ckpt/zero_to_fp32.py ./sample_ul2_ckpt/ ./sample_ul2_ckpt/0/adapter_model.bin

Load the model using:
```
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

peft_model_id = 'checkpointfolder'
base_model_id = 'Google\flan-t5-large'

model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id) # The original model path
model = PeftModel.from_pretrained(model, peft_model_id) # The fine tunned model path
```

Merge the model using
 ```
python merge_weights.py --peft_model_path=./sample_ul2_ckpt/0/adapter_model.bin --save_path=merged_model
 ```