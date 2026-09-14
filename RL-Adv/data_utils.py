import random
import json
import torch
import numpy as np
from datasets import Dataset
# NOTE: no trl import here (the old imdb `build_dataset` template that used trl.core.LengthSampler
# was removed). ppo_core.py samples generation length with a plain random.randint.

def collator(data):
    """
    Data collator function to prepare batches.
    
    Args:
        data: List of data samples
        
    Returns:
        dict: Collated data batch
    """
    if len(data) == 0:
        raise ValueError("Received empty data for collation.")
    keys = data[0].keys()
    collated_data = {key: [d[key] for d in data] for key in keys}
    return collated_data

def json_to_string(data, indent=0):
    """
    Convert JSON data to a formatted string.
    
    Args:
        data: JSON data (dict or list)
        indent: Indentation level
        
    Returns:
        str: Formatted string representation of the JSON data
    """
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

def load_http_dataset(file_path="../dataset/test2.json", sample_size=40000):
    """
    Load HTTP dataset from JSON file.
    
    Args:
        file_path: Path to the JSON file
        sample_size: Number of samples to randomly select
        
    Returns:
        Dataset: Hugging Face dataset
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    data = random.choices(data, k=sample_size)
    
    formatted_data = {
        "text": [item["Request Line"]+"\n"+json_to_string(item["Request Headers"])+"\n\n"+item["Request Body"] for item in data],
        "instruction": ["Follow these tips to generate malicious http traffic" if item["Label"]=="Malicious" else "Follow these tips to generate benign http traffic" for item in data],
        "input": [item["Request Line"] for item in data],
        "output": [item["Request Line"]+"\n"+json_to_string(item["Request Headers"])+"\n\n"+item["Request Body"] for item in data]
    }
    
    dataset = Dataset.from_dict(formatted_data)
    return dataset.shuffle(seed=61)

def text2image(all_text, img_size=(28, 28)):
    """
    Convert text data to image format.
    
    Args:
        all_text: List of text strings
        img_size: Tuple of (height, width) for the image
        
    Returns:
        list: List of image tensors
    """
    x_data = []
    for packet in all_text:
        x_data.append([ord(char) % 128 for char in packet][:img_size[0]*img_size[1]])
    padded_x_data = [row + [0] * (img_size[1]*img_size[0] - len(row)) for row in x_data]
    image_data = [torch.tensor(np.array(padded_x_data[i], dtype=np.float32).reshape(img_size[0], img_size[1])) for i in range(len(padded_x_data))]
    return image_data

def create_dataloader(dataset, batch_size=4, collate_fn=collator):
    """
    Create a dataloader from a dataset.
    
    Args:
        dataset: Hugging Face dataset
        batch_size: Batch size
        collate_fn: Collation function
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    ) 