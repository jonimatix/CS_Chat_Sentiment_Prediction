import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import datetime
import pickle
import dill

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, KFold


def get_current_datetime_num(format="%Y%m%d_%H_%M_%S"):
    return datetime.datetime.now().strftime(format)


def splitTrainTest(df, target, colforsplit):
    xtr = df.query('{} == 1'.format(colforsplit))  # .drop(target, axis = 1)
    ytr = df.query('{} == 1'.format(colforsplit))[target]
    xts = df.query('{} == 0'.format(colforsplit))  # .drop(target, axis = 1)
    return xtr, ytr, xts


def combineTrainTest(train, test):
    df = pd.concat([train, test], axis=0)
    return df


def find_optimal_rocauc_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
    """

    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr-(1-fpr), index=i),
                        'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()


def save_class_obj(path, obj, name):
    with open(path + name + '.pkl', 'wb') as f:
        dill.dump(obj, f)


def load_class_obj(path, name):
    with open(path + name + '.pkl', 'rb') as f:
        return dill.load(f)


def save_obj(path, obj, name):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path, name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def chunker(seq, size):
    ''' Splits the data frame into chunks of size N '''
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


'''
Usage: 
df = pd.DataFrame(np.random.rand(14,4), columns=['a', 'b', 'c', 'd'])

for i in chunker(df2, 6): # Chunk of size n
    print(i)
'''


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "Please input a DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep].astype(np.float64)


'''
np.isnan(X_train).sum().sort_values()
np.where(np.isnan(X_train))
np.nan_to_num(X_train)

np.any(np.isnan(X_train))
np.all(np.isfinite(X_train))
np.where(np.isfinite(X_train))

X_train[np.isfinite(X_train)]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(999, inplace=True)
'''


def best_threshold_f1score(y_true, pred_proba, proba_range, verbose=False):
    """
    Function to find the probability threshold that optimises the f1_score
    Comment: this function is not used in this excercise, but we include it in
    case the reader finds it useful
    Parameters:
    -----------
    y_true: numpy.ndarray
            array with the true labels
    pred_proba: numpy.ndarray
            array with the predicted probability
    proba_range: numpy.ndarray
            range of probabilities to explore.
            e.g. np.arange(0.1,0.9,0.01)
    Return:
    -----------
    tuple with the optimal threshold and the corresponding f1_score
    """
    scores = []
    for prob in proba_range:
        pred = [int(p > prob) for p in pred_proba]
        score = f1_score(y_true, pred)
        scores.append(score)
        if verbose:
            print("INFO: prob threshold: {}.  score :{}".format(
                round(prob, 3), round(score, 5)))

    best_score = scores[np.argmax(scores)]
    optimal_threshold = proba_range[np.argmax(scores)]

    plt.plot(scores)

    return (optimal_threshold, best_score)


def lgb_f1_score(preds, lgbDataset):
    """
    Function to compute the f1_score to be used with lightgbm methods.
    Comments: output format must be:
    (eval_name, eval_result, is_higher_better)
    Parameters:
    -----------
    preds: np.array or List
    lgbDataset: lightgbm.Dataset
    """

    binary_preds = [int(p > 0.5) for p in preds]
    y_true = lgbDataset.get_label()
    # lightgbm: (eval_name, eval_result, is_higher_better)

    return 'f1', f1_score(y_true, binary_preds), True
