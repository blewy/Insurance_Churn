import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# AUC PR Curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# plot pdp plots
from sklearn.inspection import partial_dependence

# calibration plots
from sklearn.calibration import calibration_curve


# A method for saving object data to JSON file
def save_json(self, filepath):
    """
    Serialize dictionaries into json formats

            Parameters:
                    self (dictionary): object to serialize as json
                    filepath (path): path to folder where to save the object

            Returns:
                    no return
    """
    dict_ = self

    # Creat json and save to file
    json_txt = json.dumps(dict_, indent=4)
    with open(filepath, 'w') as file:
        file.write(json_txt)


# A method for loading data from JSON file
def load_json(filepath):
    """
    Load dictionaries from json formats

            Parameters:
                    filepath (path): path to folder from where to load the object

            Returns:
                    no return
    """
    with open(filepath, 'r') as file:
        dict_ = json.load(file)

    return dict_


# function to desing a schema for data
def create_schema(data, verbose=False):
    """
    Creates a data schema from a data frame

            Parameters:
                    data (dataframe): the dataframe with the data we want to create a schema
                    verbose (bool): bool that indicated if we want to print stats from the dataframe

            Returns:
                    dictionary with the data schema
    """
    schema = {}
    for feature in data.columns:
        if verbose:
            print(f"----- {feature} ----")
            print(f"{data[feature].mean()}")
            print(f"{data[feature].std()}")
            print(f"{data[feature].min()}")
            print(f"{data[feature].max()}")
            print(f"{data[feature].unique()[0:20]}")
            print(f"-----           ----")
        thisdict = {"mean": float(data[feature].mean()),
                    "std": float(data[feature].std()),
                    "min": float(data[feature].min()),
                    "max": float(data[feature].max()),
                    "values": (data[feature].unique()[0:20]).tolist(),
                    "pct_miss": float(np.round(data[feature].isnull().mean(), 3)),
                    "type": str(data[feature].dtypes)
                    }
        schema[feature] = thisdict
    return schema


def get_curve(gt, pred, target_names, curve='roc'):
    """
    Returns the sum of two decimal numbers in binary digits.

            Parameters:
                    a (int): A decimal integer
                    b (int): Another decimal integer

            Returns:
                    binary_sum (str): Binary string of the sum of a and b
    """
    for i in range(len(target_names)):
        if curve == 'roc':
            curve_function = roc_curve
            auc_roc = roc_auc_score(gt, pred)
            label = target_names[i] + " AUC: %.3f " % auc_roc
            xlabel = "False positive rate"
            ylabel = "True positive rate"
            a, b, _ = curve_function(gt, pred)
            plt.figure(1, figsize=(7, 7))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(a, b, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)
        elif curve == 'prc':
            precision, recall, _ = precision_recall_curve(gt, pred)
            # avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(gt, pred, average='weighted')
            average_precision = average_precision_score(gt, pred)
            label = target_names[i] + " Avg. Precision: %.3f " % average_precision
            plt.figure(1, figsize=(7, 7))
            plt.step(recall, precision, where='post', label=label)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1),
                       fancybox=True, ncol=1)


def plot_pdp(model, x, feature, target=False, return_pd=False, y_pct=True, figsize=(10, 9), norm_hist=True, dec=.5):
    """
    Plot partial dependence plot suing sklearn and add a bar blot with the distribuition of the observations

            Parameters:
                    model (model): A decimal integer
                    X (dataframe): Another decimal integer
                    feature (str): Another decimal integer

            Returns:
                    plot
    """
    # Get partial dependence
    pardep = partial_dependence(model, x, [feature])

    # Get min & max values
    xmin = pardep[1][0].min()
    xmax = pardep[1][0].max()
    ymin = pardep[0][0].min()
    ymax = pardep[0][0].max()

    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.grid(alpha=.5, linewidth=1)

    # Plot partial dependence
    color = 'tab:blue'
    ax1.plot(pardep[1][0], pardep[0][0], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel(feature, fontsize=14)

    tar_ylabel = ': {}'.format(target) if target else ''
    ax1.set_ylabel('Partial Dependence{}'.format(tar_ylabel), color=color, fontsize=14)

    tar_title = target if target else 'Target Variable'
    ax1.set_title('Relationship Between {} and {}'.format(feature, tar_title), fontsize=16)

    if y_pct and ymin >= 0 and ymax <= 1:
        # Display yticks on ax1 as percentages
        fig.canvas.draw()
        labels = [item.get_text() for item in ax1.get_yticklabels()]
        labels = [int(np.float(label.replace('−', '-')) * 100) for label in labels]
        labels = ['{}%'.format(label) for label in labels]
        ax1.set_yticklabels(labels)

    # Plot line for decision boundary
    ax1.hlines(dec, xmin=xmin, xmax=xmax, color='black', linewidth=2, linestyle='--', label='Decision Boundary')
    ax1.legend()

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.hist(x[feature], bins=80, range=(xmin, xmax), alpha=.25, color=color, density=norm_hist)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('Distribution', color=color, fontsize=14)

    if y_pct and norm_hist:
        # Display yticks on ax2 as percentages
        fig.canvas.draw()
        labels = [item.get_text() for item in ax2.get_yticklabels()]
        labels = [int(np.float(label.replace('−', '-')) * 100) for label in labels]
        labels = ['{}%'.format(label) for label in labels]
        ax2.set_yticklabels(labels)

    plt.show()

    if return_pd:
        return pardep


def plot_calibration_curve(y, pred, class_labels):
    """
    Plot calibration plots for classifiction models

            Parameters:
                    y (series): Aobserved values
                    pred (series): oredicted probabilities
                    class_labels:
            Returns:
                    plot

    """
    plt.figure(figsize=(20, 20))
    for i in range(len(class_labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y, pred, n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.show()


# AUC interval estimate
def bootstrap_auc(y, pred, classes, bootstraps=100, fold_size=1000):
    """
    Returns the sum of two decimal numbers in binary digits.

            Parameters:
                    y (int): A decimal integer
                    pred (int): Another decimal integer

            Returns:
                    binary_sum (str): Binary string of the sum of a and b
    """
    statistics = np.zeros((len(classes), bootstraps))

    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        # get positive examples for stratified sampling
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            # stratified sampling of positive and negative examples
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics


# AUC confidence intervals
def print_confidence_intervals(class_labels, statistics):
    """
    Returns the sum of two decimal numbers in binary digits.

            Parameters:
                    class_labels (int): A decimal integer
                    statistics (int): Another decimal integer

            Returns:
                    binary_sum (str): Binary string of the sum of a and b
    """
    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, .95, axis=1)[i]
        min_ = np.quantile(statistics, .05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df
