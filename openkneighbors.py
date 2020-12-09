from sklearn.neighbors import KNeighborsClassifier

class KNeighborsClassifierOpenSet:
    
    classifier = None
    
    def __init__(self, n_neighbors_os=5, weights_os='uniform', algorithm_os='auto', leaf_size_os=30, p_os=2, metric_os='minkowski', metric_params_os=None, n_jobs_os=None):
        
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors_os, weights=weights_os, algorithm=algorithm_os, leaf_size=leaf_size_os, p=p_os, metric=metric_os, metric_params=metric_params_os, n_jobs=n_jobs_os)
        
    def fit(self, X, y):
        self.classifier.fit(X,y)
    
    def predict(self, X):
        return self.classifier.predict(X)
    
    def score(self, X, y):
        return self.classifier.score(X,y)
    
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)