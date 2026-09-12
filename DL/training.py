import os
import gc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)

from DL.metrics import transformer_metrics, custom_metrics

# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define default model save path
MODEL_PATH = "./models/"

def train_transformer_model(model_name, model_path, train_dataset, eval_dataset, training_args):
    """Train a transformer model."""
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=transformer_metrics
    )

    # Train model
    trainer.train()
    
    # Save model
    save_path = f"{MODEL_PATH}/{model_name}"
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    # Clean up
    del model
    del tokenizer
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return save_path

def train_custom_model(model, model_name, train_dataset, eval_dataset, training_args):
    """Train a custom model."""
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_args.per_device_train_batch_size, 
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=training_args.per_device_eval_batch_size
    )
    
    # Move model to device
    model.to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    # TrainingArguments stores num_train_epochs as a float
    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        for batch in train_dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        eval_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                inputs = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                eval_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        accuracy, precision, recall, f1 = custom_metrics(np.array(all_preds), np.array(all_labels))
        print(f"Epoch {epoch + 1}, Eval Loss: {eval_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

    # Save model
    save_path = f"{MODEL_PATH}/custom_models/{model_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path + ".bin")

    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return save_path 