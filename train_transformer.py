import os
import soundfile as sf
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from datasets import load_dataset, DatasetDict, IterableDatasetDict
from transformers import TrainerCallback, AutoConfig, AutoFeatureExtractor, TrainingArguments, Trainer, AutoModelForAudioClassification
import time
from sklearn.metrics import accuracy_score, f1_score
import argparse
import torch
from dataclasses import dataclass
from typing import List, Dict, Union
import re
import random

# Paths to datasets 
guitarset10_path_sliced = "dataset/guitarset10_sliced/"
guitarset_path_rendered = "dataset/guitarset10_rendered/"
idmt_smt_sliced = "dataset/idmt_smt_sliced/"
idmt_smt_rendered = "dataset/idmt_smt_rendered/"

# Name of each label in dataset
label_names = ["overdrive", "distortion", "chorus", "flanger", "phaser", "tremolo", "reverb", "feedback delay", "slapback delay", "low boost", "low reduct", "high boost", "high reduct"]


def extract_number(file_name):
    """
    Extract the digit in filenames that are in format "0.wav", "21.wav" and so on. 
    """
    match = re.search(r'(\d+)\.wav$', file_name)
    return int(match.group(1)) if match else None


def sort_files(files):
    """
    Sort files based on the number stored in the filename of .wav files
    """
    return sorted(files, key=extract_number)


def partition(lst, n):
    """
    Divide a list into n-partitions
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]



def create_dataset(
        save_dir, 
        stream = True, 
        partition_size = 16, 
        include_testset = False
):
    """
    Returns a huggingface dataset related including the train, validation and test sets

    Parameters
    ----------
    save_dir: The directory where all data regarding a particular model is stored (AST or WAV2VEC) 
    stream: Returns a IterableDataset if True
    partition_size: How many partitions the training set is divided into (the model is only trained on a single partition every time)
    include_test: Includes the testset in the dataset dictionary if set to True
    """
    random.seed(2) #42

    # Load index and partition training_data
    print("-- Loading labels and filenames")
    limit = 8
    train_labels = torch.load(f"{guitarset_path_rendered}gen_multiFX/train/label_tensor.pt")
    valid_labels = torch.load(f"{guitarset_path_rendered}gen_multiFX/valid/label_tensor.pt")[:limit]
    path_train = f"{guitarset_path_rendered}gen_multiFX/train/audio/"
    path_valid = f"{guitarset_path_rendered}gen_multiFX/valid/audio/"
    files_train = sort_files([os.path.join(path_train, file) for file in os.listdir(path_train)])
    files_valid = sort_files([os.path.join(path_valid, file) for file in os.listdir(path_valid)])
    train_data = list(zip(files_train, train_labels))
    valid_data = list(zip(files_valid, valid_labels))

    # Load index and partition training data
    print("-- Loading index and partitioning training data")
    index_path = os.path.join(save_dir, 'index.txt')
    with open(index_path, 'r') as file:
        content = file.read().strip()
        index = int(content)
    train_partitions = partition(train_data, partition_size)
    train_data = train_partitions[index]
    print(len(train_data))

    print("-- Shuffling filenames and labels randomly")
    random.shuffle(train_data)
    random.shuffle(valid_data)

    files_train, train_labels = zip(*train_data)
    files_valid, valid_labels = zip(*valid_data)

    # Load huggingface dataset in audiofolder format
    print("-- Loading datasets")
    dataset_train = load_dataset("audiofolder", data_files=files_train, drop_labels=True, split="train", streaming=stream)
    dataset_train = dataset_train.map(lambda example, idx: {"label": train_labels[idx]}, with_indices=True)
    dataset_valid = load_dataset("audiofolder", data_files=files_valid[:limit], drop_labels=True, split="train", streaming=False)
    dataset_valid = dataset_valid.map(lambda example, idx: {"label": valid_labels[idx]}, with_indices=True)

    # If true, include testset in dataset dictionary
    if include_testset:
        print("-- Loading testset")
        test_labels = torch.load(f"{idmt_smt_rendered}gen_multiFX/train/label_tensor.pt")
        path_test = f"{idmt_smt_rendered}gen_multiFX/train/audio/"
        files_test = sort_files([os.path.join(path_test, file) for file in os.listdir(path_test)])
        test_data = list(zip(files_test, test_labels))
        test_partitions = partition(test_data, 3)
        test_data = test_partitions[2]
        files_test, test_labels = zip(*test_data)
        dataset_test = load_dataset("audiofolder", data_files=files_test, drop_labels=True, split="train", streaming=False)
        dataset_test = dataset_test.map(lambda example, idx: {"label": test_labels[idx]}, with_indices=True)

    # Initialize dataset dictionary
    if stream:
        if include_testset:
            dataset = IterableDatasetDict({
            'train': dataset_train,
            'valid': dataset_valid,
            'test': dataset_test
            })
        else:
            dataset = IterableDatasetDict({
            'train': dataset_train,
            'valid': dataset_valid
            })        
    else:
        dataset = DatasetDict({
        'train': dataset_train,
        'valid': dataset_valid
        })
    
    # Create label2id and id2label dictionaries
    label2id, id2label = dict(), dict()
    for i, label in enumerate(label_names):
        label2id[label] = str(i)
        id2label[str(i)] = label

    print("-- Finished loading datasets")
    
    return dataset, label2id, id2label, len(train_data), index


class MultilabelTrainer(Trainer):
    """
    An extension of the huggingface 'Trainer' class where compute_loss is adjusted to work for mulit-label computations
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        inputs = inputs["input_values"]
        outputs = model(inputs)
        logits = outputs.get('logits')
        self.optimizer   
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

@dataclass
class MultiLabelCollator:
    """
    The MultiLabelCollator defines how batches are formed. Since all input are of the same length, only
    input values and labels are defined. 
    """
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch_input_values = torch.stack([feature["input_values"] for feature in features])
        batch_labels = torch.stack([feature["label"] for feature in features])
            
        batch = {
            "input_values": batch_input_values,
            "labels": batch_labels,
        }
        return batch
    
def train_evaluate(
        dataset,
        label2id, 
        id2label, 
        train_length, 
        model_checkpoint, 
        save_dir, 
        index, 
        lr_decay_rate = 0.985, 
        batch_size = 3, 
        from_checkpoint = False, 
        training_steps = None, 
        save_model = True, 
        partition_size = 16, 
        train_model = True, 
        evaluate_model = True, 
        get_preds = False, 
        preds_dataset = "valid", 
        checkpoint_str = "model_checkpoint"
):
    """
    The function defining loading the transformer model and tokenizer from a given checkpoint and trains this model on a given
    dataset. Firstly, the data is transformed to the input format given the tokenizer. Moreover, the model loaded and the
    classification head is adjusted to account for 13 classes. Furthermore, the training arguments and trainer are defined, and lastly,
    the model is trained and evaluated

    Parameters
    ----------
    dataset: A huggingface dataset including the training, validation and test set
    label2id: A dictionary mapping the labels to their respective ID
    id2label: A dictionary mapping the IDs to their respective labels
    train_length: The length of the training set
    model_checkpoint: The location to load the model and tokenizer
    save_dir: The directory to save the model weights and evaluation metrics related to the model
    index: The index of the partition that is currently being trained on
    lr_decay_rate: The rate in which the learning rate should decay after the model is trained on a partition
    batch_size: The size of each batch
    from_checkpoint: If True, the model should be trained from a checkpoint stored locally
    training_steps: How many steps to train on
    save_model: Saves the model to the save_dir if True
    partition_size: The number of partitions the training set is divided into
    train_model: If True, the model should be trained
    evaluate_model: If True, the model should be evaluated
    get_preds: If True, the model should perform predictions on a dataset and store these in the 'save_dir/predictions' folder
    preds_dataset: Defines which dataset to perform the predictions (train, valid or test)
    checkpoint_str: Defines the name of the local model checkpoint (model_checkpoint or best_model_checkpoint) 
    """

    # Define constants
    output_directory = "./models"
    num_labels = len(label_names)
    max_duration = 5.0

    # Change to local checkpoint if from_checkpoint is True
    if from_checkpoint:
        model_checkpoint = os.path.join(save_dir, checkpoint_str)

    # Load tokenizer
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    # Function defining how inputs should be processed given the tokenizer
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=feature_extractor.sampling_rate, 
            max_length=int(feature_extractor.sampling_rate * max_duration), 
            truncation=True, 
        )
        return inputs
    
    # Transform the dataset to the tokenizer format
    encoded_dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True)

    # Load model and adjust classification head
    config = AutoConfig.from_pretrained(
        model_checkpoint
    )
    config.num_labels = len(label_names)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.lm_head = nn.Sequential(
        nn.Linear(config.hidden_size, config.num_labels),
        nn.Sigmoid()
    )

    # Define training and accumilation steps
    accum_steps = 16 // batch_size
    if training_steps is None:
        total_training_steps = int(train_length / (batch_size * accum_steps))
        print(total_training_steps)
    else:
        total_training_steps = training_steps

    #Read learning rate from file
    lr_path = os.path.join(save_dir, 'lr_rate.txt')
    with open(lr_path, 'r') as file:
        content = file.read().strip()
        learning_rate = float(content)

    # Define the training arguments 
    args = TrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accum_steps,  
        evaluation_strategy="no",
        save_strategy="steps" if save_model else "no",
        load_best_model_at_end=False,
        logging_dir='./logs',
        learning_rate=learning_rate,
        logging_steps=20,
        save_steps=500,  
        save_total_limit=1 if save_model else None,  
        max_steps=total_training_steps,
        lr_scheduler_type='constant',
        fp16=True
    )

    # Function computing accuracy and f1-scores during evaluation
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        logits, labels = eval_pred
        probabilities = np.array(torch.sigmoid(torch.from_numpy(logits)))
        predictions = (probabilities > 0.5).astype(int)
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)
        f1_micro = f1_score(y_true=labels, y_pred=predictions, average='micro')
        f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')
        return {"accuracy": accuracy, "f1_micro": f1_micro, "f1_macro": f1_macro}
    
    # Define collator and callback logging metrics to logs.txt
    data_collator = MultiLabelCollator()
    class LoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            with open(os.path.join(save_dir, 'logs.txt'), 'a') as log_file:
                log_file.write(f"Step: {state.global_step}, {logs}\n")
    
    # Define the multi-label trainer with arguments 
    trainer = MultilabelTrainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset['train'].with_format("torch"),  
        eval_dataset=encoded_dataset['valid'].with_format("torch"), 
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        callbacks=[LoggingCallback],
    )

    # Load state of trainer
    trainer_state = os.path.join(save_dir, "trainer_state.pt")
    if from_checkpoint:
        trainer.state = torch.load(trainer_state)

    # Train the model if True
    if train_model:
        print("--Training the model: ")
        start = time.time()
        if from_checkpoint:
            trainer.train(resume_from_checkpoint=model_checkpoint)
        else:
            trainer.train()
        end = time.time()
        print(f"    --Total time of training: {end - start}")

    # Evaluate the model if True
    if evaluate_model:
        print("--Evaluating the model: ")
        evaluation = trainer.evaluate()   
        print(f"    --{evaluation}")

        print(f"State: {trainer.state.log_history}")

    # Save the model, if evaluation f1 is higher
    if save_model:
        # Update index
        index_path = os.path.join(save_dir, 'index.txt')
        with open(index_path, 'w') as file:
            new_index = (index + 1) % partition_size
            file.write(f"{new_index}")

        # Read current and best f1
        eval_f1 = evaluation["eval_f1_micro"]
        best_f1_path = os.path.join(save_dir, 'best_f1.txt')
        with open(best_f1_path, 'r') as file:
            content = file.read().strip()
            best_f1 = float(content)

        # Save model seperately if current f1 is better than previous best f1
        if eval_f1 > best_f1:
            print(f"--Evaluation F1 score were better than previous (saving model)")
            trainer.save_model(os.path.join(save_dir, "model_checkpoint_best"))
            torch.save(trainer.state, os.path.join(save_dir, "trainer_state_best.pt"))
            with open(best_f1_path, 'w') as file:
                file.write(f"{eval_f1}")
        else:
            print(f"--Evaluation F1 score are worse than previous (not saving model)")
        print(f"    --Current F1: {eval_f1}, previous best-F1: {best_f1}")

        # Save model and training state
        print("    --Saving model: ")
        trainer.save_model(os.path.join(save_dir, "model_checkpoint"))
        torch.save(trainer.state, trainer_state)
        with open(os.path.join(save_dir, "total_steps.txt"), 'a') as file:
            file.write(f"{total_training_steps}\n")
        with open(lr_path, 'w') as file:
                file.write(f"{learning_rate * lr_decay_rate}")
        print("    --Model saved successfully")

    # Make predictions on a datasets and save these if True
    if get_preds:
        print(f"--Making prediction on the model using the {preds_dataset}-dataset")
        predictions_output = trainer.predict(encoded_dataset[preds_dataset].with_format("torch"))
        logits = predictions_output.predictions
        threshold = 0.5
        sigmoid_logits = torch.sigmoid(torch.tensor(logits)).numpy()
        predictions = (sigmoid_logits > threshold).astype(int)
        print("-- Saving predictions")
        folder_path = f"{save_dir}/predictions"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.savetxt(os.path.join(folder_path, f'{preds_dataset}_actual_3.txt'), predictions_output.label_ids, fmt='%d')
        np.savetxt(os.path.join(folder_path, f'{preds_dataset}_predicted_3.txt'), predictions, fmt='%d')

    return model

def parse_arguments():
    """
    Parsing all the arguments given in the terminal

    Arguments
    ----------
    dataset: Define the dataset to train on ('small' or 'standard')
    stream: Whether to use streaming datasets ('True' or 'False')
    model: Which model to train on ('ast' or 'wav2vec')
    train: Whether to train the model or not ('True' or 'False')
    predict: Whether to make predictions on a dataset or which dataset to make predictions on ('no', 'train', 'valid' or 'test')
    """
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--dataset', type=str, help='Size of dataset to train with', default="standard")
    parser.add_argument('--stream', type=str, help='Whether to use map-based datasets or iterable datasets', default="True")
    parser.add_argument('--model', type=str, help='Which model to train', default="ast")
    parser.add_argument('--train', type=str, help='Whether to train the model or not', default="True")
    parser.add_argument('--predict', type=str, help='Whether to predict on a dataset, and which dataset to predict on', default="no")
    args = parser.parse_args()
    return args

def parse_bool(argument):
    if argument == "True":
        return True
    else:
        return False

if __name__ == "__main__":
    # Empty cuda cache before training:
    torch.cuda.empty_cache() 

    # Parse arguments
    args = parse_arguments()
    if args.dataset == "small":
        guitarset10_path_sliced = "dataset/guitarset10_sliced_small/"
        guitarset_path_rendered = "dataset/guitarset10_rendered_small/" 
    stream = parse_bool(args.stream)
    train = parse_bool(args.train)

    if args.stream == "True":
        stream = True
    else: 
        stream = False
    if args.train == "True":
        train = True
    else:
        train = False
    
    if args.model == "wav2vec":
        model_checkpoint = "facebook/wav2vec2-base"
        save_dir = "WAV2VEC"
        best_checkpoint = "WAV2VEC/model_checkpoint_best"
    elif args.model == "ast":
        model_checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
        save_dir = "AST"
        best_checkpoint = "AST/model_checkpoint_best"

    # Train the model
    if train:
        while True:
            dataset, label2id, id2label, train_len, index = create_dataset(save_dir, stream=stream)
            train_evaluate(dataset, label2id, id2label, train_len, model_checkpoint, save_dir, index, from_checkpoint=True, training_steps=None, save_model=True, lr_decay_rate = 1)
    
    # Save predictions
    if args.predict != "no":
        dataset, label2id, id2label, train_len, index = create_dataset(save_dir, stream=stream, include_testset=True)
        train_evaluate(dataset, label2id, id2label, train_len, model_checkpoint, save_dir, index, from_checkpoint=True, save_model=False, lr_decay_rate = 1, train_model=False, checkpoint_str="model_checkpoint_best", evaluate_model=False, get_preds=True, preds_dataset=args.predict)