import os
import torch

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 设置代理环境变量
# os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'


major_version, minor_version = torch.cuda.get_device_capability()



from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

import os
import requests

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit", # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit", # Instruct version of Gemma 2b
    "unsloth/llama-3-8b-bnb-4bit", # [NEW] 15 Trillion token Llama-3
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)



model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)



alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
# dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
# dataset = dataset.map(formatting_prompts_func, batched = True,)





#加载自己的http数据集
import json
with open("../dataset/train_data2.json","r") as f:
  data = json.load(f)
from datasets import Dataset
def json_to_string(data, indent=0):
    result = []
    indent_str = ' ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result.append(f'{indent_str}{key}:')
                result.append(json_to_string(value, indent + 1))
            else:
                result.append(f'{indent_str}{key}: {value}')
    elif isinstance(data, list):
        for item in data:
            result.append(json_to_string(item, indent))
    else:
        result.append(f'{indent_str}{data}')
    return '\n'.join(result)

#benign_instruction = "Follow the prompts to generate benign HTTP traffic with request lines, request headers, and payloads, for example: GET /?FNKM=GMUEZPM HTTP/1.1\nCache_Control: no-cache\n\n\{PAYLOAD\}"
#malicious_instruction = "Follow the prompts to generate malicious HTTP traffic with request lines, request headers, and payloads, for example: GET /?FNKM=GMUEZPM HTTP/1.1\nCache_Control: no-cache\n\n\{PAYLOAD\}"

formatted_data = {
                "text": [item["Request Line"]+"\n"+json_to_string(item["Request Headers"])+"\n\n"+item["Request Body"] for item in data],
                "instruction": ["Follow these tips to generate malicious http traffic" if item["Label"]=="Malicious" else "Follow these tips to generate benign http traffic" for item in data],
                "input" : [item["Request Line"] for item in data],
                "output":[item["Request Line"]+"\n"+json_to_string(item["Request Headers"])+"\n\n"+item["Request Body"] for item in data]
}
dataset = Dataset.from_dict(formatted_data)
dataset = dataset.shuffle(seed=42)




from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 设置环境变量以禁用 NCCL 中的 P2P 和 IB
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = dataset.select(range(min(100, len(dataset)))),
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 64,
        warmup_steps = 5,
        max_steps = 500,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "../model/lamma_outputs",
        save_strategy="steps",
        save_steps=100,  # 每500步保存一次模型
        save_total_limit=2
    ),
)


trainer_stats = trainer.train()

