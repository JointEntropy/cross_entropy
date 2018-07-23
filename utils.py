import pickle
import os
import numpy as np
import json
from scipy.sparse import csr_matrix
import time
from sklearn.metrics import accuracy_score

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def inverse_ohe(ohe_outputs, ohe_encoder):
    return ohe_encoder.active_features_[ohe_outputs.argmax(axis=1)]


def blend(models, X, weights=None, proba=False):
    preds = []
    weights = weights or [1/len(models), ] * len(models)
    for model, weight in zip(models, weights):
        preds.append(model.predict_proba(X)* weight)
    if proba:
        return np.stack(preds).sum(axis=0)
    else:
        return np.stack(preds).sum(axis=0).argmax(axis=1)

def blending_weights_search(models, data, labels, limit=10**10,
                          print_every = 100, accuracy=10**2):
    def partitions(n,k,l=1):
        if k < 1:
            raise StopIteration
        if k == 1:
            if n >= l:
                yield (n,)
            raise StopIteration
        for i in range(l,n+1):
            for result in partitions(n-i,k-1,i):
                yield (i,)+result    
    best = (0, None)
    for i, weights in  zip(range(limit), map(lambda x: [t/accuracy for t in x], partitions(accuracy, k=4, l=10))):
        te_score = accuracy_score(labels, blend(models, data, weights))
        if i % print_every == 0:
            print(i, best)
        if te_score> best[0]:
            best = (te_score, weights)
    return best

def multiclass_auc(y_true, y_preds, metric=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if metric == 'rocauc':
        metric = roc_auc_score
    elif metric == 'pc':
        metric = average_precision_score
    scores = []
    for c in set(y_true):
        true = y_true == c
        preds = y_preds == c
        scores.append(metric(true, preds))
    return np.array(scores).mean()
