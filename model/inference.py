import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, EvalPrediction

from typing import Tuple, Optional, List

import numpy as np

from tqdm import tqdm

import os
import json
import yaml

from preprocessor import Preprocessor
from dataset import CustomDataset
from trainer_qa import SpacingTrainer
from utils_qa import compute_metrics, post_process_function
from model import CustomTokenClassification

if __name__ == "__main__":
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    model = AutoModelForTokenClassification.from_pretrained(
        config['model_path'], 
        from_tf=bool(".ckpt" in config['model_path']),
        num_labels=4
    )

    # model = CustomTokenClassification(config['model_name'], num_labels=4)
    # model.load_state_dict(torch.load(os.path.join(config['model_path'], 'model.pt')))

    model.to(device)

    preprocessor = Preprocessor(config['max_len'], tokenizer)
    test_dataset = CustomDataset(data_path=config['test_data_path'], transform=preprocessor.get_input_features)

    training_args = TrainingArguments(
        output_dir='./results'
    )

    trainer = SpacingTrainer(
        model=model,                         
        args=training_args,                  
        train_dataset=None,         
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        post_process_function=post_process_function
    )

    prediction = trainer.predict(test_dataset=test_dataset, test_examples=test_dataset)
    print(prediction["metrics"])