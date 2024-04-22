from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
 

def parse_transformer_metrics(path, metric):
    """
    Parse a metric from a logs.txt file
    """
    metrics = []

    pattern_str = f"'{metric}': " + r"(\d+\.\d+)"
    pattern = re.compile(pattern_str)

    with open(path, 'r') as log_file:
        log_lines = log_file.readlines()
        for line in log_lines:
            pattern_match = pattern.search(line)            
            if pattern_match:
                metrics.append(float(pattern_match.group(1)))

    return metrics


def get_transformer_predictions(folder_path, dataset):
    """
    Return actual and predicted instanced on a given dataset
    """
    loaded_valid_actual = np.loadtxt(os.path.join(folder_path, f'{dataset}_actual.txt')).astype(int)
    loaded_valid_predicted = np.loadtxt(os.path.join(folder_path, f'{dataset}_predicted.txt')).astype(int)
    return loaded_valid_actual.tolist(), loaded_valid_predicted.tolist()


def visualize_confusion_matrix(data, f1_scores):
    """
    Plot a multi-label confusion matrix
    """
    tags = list(data.keys())

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(5, 6))

    for i in range(13, 16):
        fig.delaxes(axes.flatten()[i])

    for ax, tag, f1_score in zip(axes.flatten(), tags, f1_scores):
        conf_mat = data[tag]
        
        ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.6)
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                if conf_mat[i, j] > 0:
                    ax.text(x=j, y=i, s=conf_mat[i, j], 
                            va='center', ha='center', size='x-small')

        ax.set_title(f'F1: {f1_score:.2f}', fontsize=7, pad=2)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel(tag, fontsize=7, fontweight='bold')

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def visualize_f1_curve(f1_scores, steps_per_epoch=16):
    """
    Plot F1 curves over a certain number of epochs
    """
    plt.figure(figsize=(10, 6))  
    epochs = [step / steps_per_epoch for step in range(1, len(f1_scores) + 1)]
    
    plt.plot(epochs, f1_scores, label='F1-score', color="#B33C3C")

    plt.xlabel('Epoch', fontdict={'size': 12, 'weight': 'bold'}, fontsize=12)  
    plt.ylabel('F1-score', fontdict={'size': 12, 'weight': 'bold'}, fontsize=12)

    plt.grid(True)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)

    plt.show()
    

def visualize_loss(train_losses, validation_losses, steps_per_epoch=16):
    """
    Plot training loss and validation loss over a certain number of epochs
    """
    plt.figure(figsize=(10, 6)) 
    epochs = [step / steps_per_epoch for step in range(1, len(train_losses) + 1)]
    
    plt.plot(epochs, train_losses, label='Training Loss', color="cornflowerblue")
    plt.plot(epochs, validation_losses, label='Validation Loss', color="seagreen")

    plt.xlabel('Epoch', fontdict={'size': 12, 'weight': 'bold'}, fontsize=12)  
    plt.ylabel('Loss', fontdict={'size': 12, 'weight': 'bold'}, fontsize=12)

    plt.legend(prop={'size': 12}) 

    plt.grid(True)

    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)

    plt.show()


def benchmark(scores):
    """
    Plot graph displaying comparing F1-micro and F1-macro scores on the validation set (guitarset) and test set (idmt-smt-guitar).
    The models compared are resnet18, CRNN, Audio Spectrogram Transformer (ast) and wav2vec-2
    """
    models = list(scores.keys())
    micro_F1_guitarSet10 = [score[0] for score in scores.values()]
    macro_F1_guitarSet10 = [score[1] for score in scores.values()]
    micro_F1_IDMT = [score[2] for score in scores.values()]
    macro_F1_IDMT = [score[3] for score in scores.values()]

    barWidth = 0.2
    r1 = np.arange(len(models))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    plt.figure(figsize=(12, 8))

    bars1 = plt.bar(r1, micro_F1_guitarSet10, color='cornflowerblue', width=barWidth, edgecolor='white', label='micro_F1_GuitarSet10')
    bars2 = plt.bar(r2, macro_F1_guitarSet10, color='lightsteelblue', width=barWidth, edgecolor='white', label='macro_F1_GuitarSet10')
    bars3 = plt.bar(r3, micro_F1_IDMT, color='seagreen', width=barWidth, edgecolor='white', label='micro_F1_IDMT')
    bars4 = plt.bar(r4, macro_F1_IDMT, color='mediumseagreen', width=barWidth, edgecolor='white', label='macro_F1_IDMT')

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), ha='center', va='bottom') 

    plt.title('Micro and Macro F1 Scores for Models')
    plt.xlabel('Benchmark Model', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(models))], models)
    plt.legend()
    plt.show()


def calculate_confusion_matrices(actuals, predictions, labels):
    """
    Calculate confusion matrices for a multi-label classification
    """
    
    # Calculate confusion matrix using scikit-learn for each label
    cms = multilabel_confusion_matrix(actuals, predictions)
    
    # Pack into a dictionary with labels as keys
    cm_dict = {label: cm for label, cm in zip(labels, cms)}
    
    return cm_dict


def f1_score_per_label(y_true, y_pred, labels):
    """
    Calculates F1 score per label
    """
    # Convert labels to binary format for multi-label classification
    y_true_binary = label_binarize(y_true, classes=labels)
    y_pred_binary = label_binarize(y_pred, classes=labels)

    # Calculate F1 scores per label
    scores = {}
    for i, label in enumerate(labels):
        score = f1_score(y_true_binary[:, i], y_pred_binary[:, i], zero_division=0)
        scores[label] = score

    return scores


if __name__ == "__main__":
    # Names of all 13 labels
    label_names = ["overdrive", "distortion", "chorus", "flanger", "phaser", "tremolo", "reverb", "feedback delay", "slapback delay", "low boost", "low reduct", "high boost", "high reduct"]

    # AST metrics
    ast_logs_path = "AST/logs.txt"
    ast_train_losses = parse_transformer_metrics(ast_logs_path, "train_loss")
    ast_valid_losses = parse_transformer_metrics(ast_logs_path, "eval_loss")
    ast_f1_micro = parse_transformer_metrics(ast_logs_path, "eval_f1_micro")
    ast_f1_macro = parse_transformer_metrics(ast_logs_path, "eval_f1_macro")
    ast_actual, ast_predicted = get_transformer_predictions("AST/predictions", "valid")

    # Benchmark scores
    scores_test = {
    "resnet18": (0.85, 0.80, 0.88, 0.84),
    "CRNN": (0.80, 0.75, 0.90, 0.82),
    "audio_spectrogram_transformer": (0.83, 0.79, 0.87, 0.82),
    "wav2vec": (0.87, 0.83, 0.91, 0.86),
    }   

    # Compute multi-label confusion matrix for AST
    f1_dict = f1_score_per_label(ast_actual, ast_predicted, label_names)
    matrices = calculate_confusion_matrices(ast_actual, ast_predicted, label_names)

    # Plot loss and f1 for AST
    visualize_confusion_matrix(matrices, f1_dict.values())
    visualize_loss(ast_train_losses, ast_valid_losses)
    visualize_f1_curve(ast_f1_micro)

    # Plot benchmark
    benchmark(scores_test)

    
