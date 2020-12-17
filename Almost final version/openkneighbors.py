from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import numpy as np

class KNeighborsClassifierOpenSet:
    
    classifier = None
    threshold = 0.6
    
    def __init__(self, n_neighbors_os=5, weights_os='uniform', algorithm_os='auto', leaf_size_os=30, p_os=2, metric_os='minkowski', metric_params_os=None, n_jobs_os=None):
        
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors_os, weights=weights_os, algorithm=algorithm_os, leaf_size=leaf_size_os, p=p_os, metric=metric_os, metric_params=metric_params_os, n_jobs=n_jobs_os)
        
    def fit(self, X, y):
        self.classifier.fit(X,y)
        
    def predict(self, X):
            return self.classifier.predict(X)               
    
    def predict_open(self, X):
        probs = self.classifier.predict_proba(X)
        result = np.argmax(probs, axis=1)
        for i in range(probs.shape[0]):
            if np.max(probs[i,:]) < self.threshold:
                result[i] = -1
        return result        
    
    def score(self, X, y):
        return self.classifier.score(X,y)
    
    def score_open(self, X, y):
        predictions = self.predict_open(X)
        is_correct = (predictions == y)
        return np.count_nonzero(is_correct)/X.shape[0]
    
    def f1_score(self, X, y):
        predictions = self.predict(X)
        return f1_score(y, predictions, average='micro')
    
    def f1_score_open(self, X, y):
        predictions = self.predict_open(X)
        return f1_score(y, predictions, average='micro')    
    
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
    
    def set_threshold(self, trsh):
        self.threshold = trsh