from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
 

def visualize_confusion_matrix(data, f1_scores):
    tags = list(data.keys())

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(14, 14))

    # Remove the subplots which are not used
    for i in range(13, 16):
        fig.delaxes(axes.flatten()[i])

    for ax, tag, f1_score in zip(axes.flatten(), tags, f1_scores):
        conf_mat = data[tag]
        
        # Show the confusion matrix
        ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                # Only annotate cells with numbers other than 0
                if conf_mat[i, j] > 0:
                    ax.text(x=j, y=i, s=conf_mat[i, j], 
                            va='center', ha='center', size='xx-large')

        # Title with F1 score
        ax.set_title(f'F1: {f1_score:.2f}', fontsize=9, pad=2)

        # Disable x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Label below the matrix
        ax.set_xlabel(tag, fontsize=9, fontweight='bold')

    # Adjust layout to fit everything more tightly
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    plt.show()
    
 

def visualize_loss(train_losses, validation_losses):
    plt.figure(figsize=(10, 6))  # Optionally set the figure size
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', color = 'mediumseagreen')
    plt.plot(epochs, validation_losses, label='Validation Loss', color = 'cornflowerblue')

    # Add a title and labels to the axes.
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Add a legend to clarify which line is which.
    plt.legend()

    # Optionally set a grid and the style of the plot
    plt.grid(True)
    plt.style.use('ggplot')

    # Display the plot.
    plt.show()

def benchmark(scores):
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

    # Adding the scores on top of the bars
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
    
    Parameters:
    - actuals: a list or a numpy array of actual label vectors
    - predictions: a list or a numpy array of predicted label vectors
    - labels: a list of class labels
    
    Returns:
    - cm_dict: a dictionary with class labels as keys and corresponding confusion matrix as values
    """
    
    # Calculate confusion matrix using scikit-learn for each label
    cms = multilabel_confusion_matrix(actuals, predictions)
    
    # pack into a dictionary with labels as keys
    cm_dict = {label: cm for label, cm in zip(labels, cms)}
    
    return cm_dict

def f1_score_per_label(y_true, y_pred, labels):
    """
    Calculates F1 score per label
    
    Parameters:
    y_true: array-like of shape (n_samples,)
        True labels.
    
    y_pred: array-like of shape (n_samples,)
        Predicted labels.
    
    labels: array-like of shape (n_classes,)
        An array of all possible labels.
    
    Returns:
    scores: dict
        A dictionary with labels as keys and their F1 score as values.
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
    # Declare actual and predicted labels + label names
    actual_test = [
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], 
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1], 
    [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    predicted_test = [
    [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0], 
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0], 
    [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0], 
    [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]]
    training_loss_test = [2.3, 1.98, 1.76, 1.62, 1.4, 1.3, 1.2, 1.1, 1.05, 1.0]
    validation_loss_test = [2.5, 2.2, 2.0, 1.8, 1.8, 1.7, 1.6, 1.6, 1.55, 1.5]
    scores_test = {
    "resnet18": (0.85, 0.80, 0.88, 0.84),
    "CRNN": (0.80, 0.75, 0.90, 0.82),
    "audio_spectrogram_transformer": (0.83, 0.79, 0.87, 0.82),
    "wav2vec": (0.87, 0.83, 0.91, 0.86),
    }   
    label_names = ["overdrive", "distortion", "chorus", "flanger", "phaser", "tremolo", "reverb", "feedback delay", "slapback delay", "low boost", "low reduct", "high boost", "high reduct"]

    # Calculate f1 per label and confusion matrix
    f1_dict = f1_score_per_label(actual_test, predicted_test, label_names)
    matrices = calculate_confusion_matrices(actual_test, predicted_test, label_names)
    #TODO: Visualize confusion matrix
    visualize_confusion_matrix(matrices, f1_dict.values())

    # Make a visualization of benchmark table (microF1 and macroF1 for guitarset10 and IDMT)
    #TODO: Visualize benchmark
    benchmark(scores_test)

    # Visualize validation and training loss
    visualize_loss(training_loss_test, validation_loss_test)