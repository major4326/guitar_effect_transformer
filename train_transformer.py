import os
import soundfile as sf
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
import torchaudio
from datasets import Dataset, load_dataset, DatasetDict, Features, IterableDatasetDict
from evaluate import load
from transformers import ASTModel, TrainerCallback, TrainerState, AutoConfig, AutoFeatureExtractor, ASTForAudioClassification, TrainingArguments, Trainer, ASTConfig, AutoModelForAudioClassification
from functools import partial
import time
from sklearn.metrics import accuracy_score, f1_score
import argparse
from datasets.features import Audio
import torch
from dataclasses import dataclass
from typing import List, Dict, Union
import re
from torch.optim import AdamW
import random

torch.cuda.empty_cache() 

guitarset10_path = "dataset/"
guitarset10_path_sliced = "dataset/guitarset10_sliced/"
guitarset_path_rendered = "dataset/guitarset10_rendered/"

def gen(path):
    labels = torch.load(os.path.join(path, "label_tensor.pt"))
    audio_path = os.path.join(path, "audio")
    sorted_filenames = sorted(os.listdir(audio_path), key=lambda filename: int(filename.split('.')[0]))
    for index, filename in enumerate(sorted_filenames):
        if filename.lower().endswith('.wav'):
            full_path = os.path.join(audio_path, filename)
            waveform, sampling_rate = torchaudio.load(full_path)
            waveform = waveform.squeeze().numpy()

            yield {
                'audio': {
                    'path': full_path,
                    'array': np.array(waveform),
                    'sampling_rate': sampling_rate},
                'label': labels[index]
            }

label_names = ["overdrive", "distortion", "chorus", "flanger", "phaser", "tremolo", "reverb", "feedback delay", "slapback delay", "low boost", "low reduct", "high boost", "high reduct"]
metric = load("accuracy")

def extract_number(file_name):
    match = re.search(r'(\d+)\.wav$', file_name)
    return int(match.group(1)) if match else None

def sort_files(files):
    return sorted(files, key=extract_number)

def create_dataset(stream = True):
    random.seed(42)

    print("-- Loading labels")
    limit = 80000
    train_labels = torch.load(f"{guitarset_path_rendered}gen_multiFX_03262024/train/label_tensor.pt")
    valid_labels = torch.load(f"{guitarset_path_rendered}gen_multiFX_03262024/valid/label_tensor.pt")[:limit]
    path_train = f"{guitarset_path_rendered}gen_multiFX_03262024/train/audio/"
    path_valid = f"{guitarset_path_rendered}gen_multiFX_03262024/valid/audio/"

    print("-- Loading filenames")

    files_train = sort_files([os.path.join(path_train, file) for file in os.listdir(path_train)])
    files_valid = sort_files([os.path.join(path_valid, file) for file in os.listdir(path_valid)])

    print("-- Shuffling filenames and labels randomly")
    train_data = list(zip(files_train, train_labels))
    valid_data = list(zip(files_valid, valid_labels))

    random.shuffle(train_data)
    random.shuffle(valid_data)

    files_train, train_labels = zip(*train_data)
    files_valid, valid_labels = zip(*valid_data)

    print("-- Loading datasets")
    dataset_train = load_dataset("audiofolder", data_files=files_train, drop_labels=True, split="train", streaming=stream)
    dataset_train = dataset_train.map(lambda example, idx: {"label": train_labels[idx]}, with_indices=True)
    dataset_valid = load_dataset("audiofolder", data_files=files_valid[:limit], drop_labels=True, split="train", streaming=False)
    dataset_valid = dataset_valid.map(lambda example, idx: {"label": valid_labels[idx]}, with_indices=True)

    if stream:
        dataset = IterableDatasetDict({
        'train': dataset_train,
        'valid': dataset_valid
        })
    else:
        dataset = DatasetDict({
        'train': dataset_train,
        'valid': dataset_valid
        })
    
    
    label2id, id2label = dict(), dict()
    for i, label in enumerate(label_names):
        label2id[label] = str(i)
        id2label[str(i)] = label

    print("-- Finished loading datasets")
    
    return dataset, label2id, id2label, len(files_train)

class MultilabelTrainer(Trainer):

    """def create_optimizer(self):
        # Split parameters between base transformer layer and classifier layer
        # Usually the output layer is called `classifier`, `lm_head`, etc. depending on the model
        # You will need to adjust this according to the actual architecture you are using
        # For example, you may have 'output' instead of 'classifier'
        classifier = 'lm_head.weight'
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if classifier not in n],
                'lr': 3e-6
            },
            {
                'params': [p for n, p in self.model.named_parameters() if classifier in n],
                'lr': 1e-3
            }
        ]

        # Adjust epsilon and other optimizer parameters as needed
        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)
        print(optimizer)
        print(type(optimizer))
        return optimizer"""

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        inputs = inputs["input_values"]
        outputs = model(inputs)
        logits = outputs.get('logits')
        self.optimizer
        
        # Replace CrossEntropyLoss with BCEWithLogitsLoss for multi-label classification
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

@dataclass
class DataCollatorForWav2Vec2:
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Batch input values and attention masks
        #print(features[0])
        #if isinstance(features[0]["input_values"], np.ndarray):
        #batch_input_values = torch.stack([torch.from_numpy(feature["input_values"]) for feature in features])
        batch_input_values = torch.stack([feature["input_values"] for feature in features])
        #batch_attention_masks = torch.stack([torch.from_numpy(feature["attention_mask"]) for feature in features])
        
        # Batch labels, assuming your labels are already tensors and have the same shape across samples
        batch_labels = torch.stack([feature["label"] for feature in features])
            

        # The resulting batch is a dictionary with the corresponding stacked tensors
        batch = {
            "input_values": batch_input_values,
            #"attention_mask": batch_attention_masks,
            "labels": batch_labels,
        }
        return batch

def train_evaluate(dataset, label2id, id2label, model_type, train_length, batch_size = 3, from_checkpoint = False, training_steps = None, save_model = True, learning_rate=1e-5):
    output_directory = "./models"

    num_labels = len(label_names)

    max_duration = 5.0
    if model_type == "wav2vec":
        model_checkpoint = "facebook/wav2vec2-base"
        #save_file = "wav2vec-finetuned-guitar" #Remove or replace?
        save_dir = "WAV2VEC2"
    elif model_type == "ast":
        model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
        #save_file = "ast-finetuned-guitar" #Remove/replace?
        save_dir = "AST"
    elif model_type == "sew-d":
        model_checkpoint = "asapp/sew-d-tiny-100k"
        save_dir = "SEWD"

    if from_checkpoint:
        model_checkpoint = os.path.join(save_dir, "model_checkpoint")

    #save_path = os.path.join(output_directory,save_file)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=feature_extractor.sampling_rate, 
            max_length=int(feature_extractor.sampling_rate * max_duration), 
            truncation=True, 
        )
        return inputs

    encoded_dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True)
    print(encoded_dataset)
    config = AutoConfig.from_pretrained(
        model_checkpoint
    )
    config.num_labels = len(label_names)

    """model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        #problem_type="multi_label_classification",
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )"""
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        #problem_type="multi_label_classification",
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Add a multi-label classification head
    model.lm_head = nn.Sequential(
        nn.Linear(config.hidden_size, config.num_labels),
        nn.Sigmoid()
    )

    accum_steps = 16 // batch_size

    if training_steps is None:
        total_training_steps = train_length // (batch_size * accum_steps)
        print(total_training_steps)
    else:
        total_training_steps = training_steps

    """
    args = TrainingArguments(
        save_file,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=3e-5,
        logging_dir=os.path.join(save_dir, "logs"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,
        per_device_eval_batch_size=batch_size,
        max_steps=total_training_steps,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_steps=100,model_checkpoint
        dataloader_drop_last=True,
    )
    """
    #pretrained_names = [f'bert.{k}' for (k, v) in model.]
    #pretrained_names = [f'bert.{k}' for (k, v) in model.bert.named_parameters()]

    #new_params= [v for k, v in model.named_parameters() if k not in pretrained_names]

    """optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if "model.lm_head" not in n],
                'lr': 3e-6
            },
            {
                'params': [p for n, p in model.named_parameters() if "model.lm_head" in n],
                'lr': 1e-3
            }]
                
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
    )"""
    
    args = TrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,  # adjust grad accumulation based on your batch size and resources
        #learning_rate=learning_rate,
        evaluation_strategy="no",
        save_strategy="steps" if save_model else "no",
        load_best_model_at_end=False,
        num_train_epochs=3,  # Adjust based on your specific needs
        logging_dir='./logs',
        learning_rate=learning_rate,
        logging_steps=1,
        save_steps=500,  # Adjust based on your needs
        save_total_limit=1 if save_model else None,  # Only save the most recent model
        # Use training_steps if provided, else calculate from dataset
        max_steps=training_steps if training_steps is not None else (train_length // batch_size) * 3,  # Multiply dataset size with epochs
    )

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        logits, labels = eval_pred
        # Transform logits to probabilities
        probabilities = np.array(torch.sigmoid(torch.from_numpy(logits)))
        # Convert probabilities to binary predictions using a threshold of 0.5
        predictions = (probabilities > 0.5).astype(int)

        # Compute metrics, accuracy, and f1 score for multi-label classification
        print("--------")
        print(labels)
        print(predictions)
        print("--------")
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        f1_micro = f1_score(y_true=labels, y_pred=predictions, average='micro')
        f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')
        # Precision
        # Recall
        return {"accuracy": accuracy, "f1_micro": f1_micro, "f1_macro": f1_macro}
    
    data_collator = DataCollatorForWav2Vec2()

    class LoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            # Do something when logging, like writing to a file
            # `logs` is a dictionary with the metrics to be logged
            with open(os.path.join(save_dir, 'logs.txt'), 'a') as log_file:
                log_file.write(f"Step: {state.global_step}, {logs}\n")
    
    trainer = MultilabelTrainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset['train'].with_format("torch"),  # Assuming dataset is a `DatasetDict`
        eval_dataset=encoded_dataset['valid'].with_format("torch"), # Assuming dataset is a `DatasetDict`
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        callbacks=[LoggingCallback],
    )

    trainer_state = os.path.join(save_dir, "trainer_state.pt")
    if from_checkpoint:
        trainer.state = torch.load(trainer_state)

    #if checkpoint is not None:
    #    checkpoint_path = os.path.join(save_file, checkpoint)
    #    state = TrainerState.load_from_json(os.path.join(checkpoint_path, "trainer_state.json"))
    #    trainer.state = state

    print("Training the model: ")
    start = time.time()
    if from_checkpoint:
        trainer.train(resume_from_checkpoint=model_checkpoint)
    else:
        trainer.train()
    end = time.time()
    print(f"Total time of training: {end - start}")
    if save_model:
        print("Saving model: ")
        trainer.save_model(os.path.join(save_dir, "model_checkpoint"))
        torch.save(trainer.state, trainer_state)
        with open(os.path.join(save_dir, "total_steps.txt"), 'a') as file:
            file.write(f"{total_training_steps}\n")
        print("Model saved successfully")

    print("Evaluating the model: ")
    evaluation = trainer.evaluate()   
    print(evaluation)
    return model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--dataset', type=str, help='Size of dataset to train with', default="standard")
    parser.add_argument('--stream', type=str, help='Whether to use map-based datasets or iterable datasets', default="True")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Use small dataset if specified in the arguments
    args = parse_arguments()
    if args.dataset == "small":
        guitarset10_path_sliced = "dataset/guitarset10_sliced_small/"
        guitarset_path_rendered = "dataset/guitarset10_rendered_small/" 
    if args.stream == "True":
        stream = True
    else: 
        stream = False

    dataset, label2id, id2label, train_len = create_dataset(stream=stream)

    # One epoch: 22453
    train_evaluate(dataset, label2id, id2label, "ast", train_len, from_checkpoint=True, training_steps=int(22453//32), save_model=True, learning_rate=3e-6)
