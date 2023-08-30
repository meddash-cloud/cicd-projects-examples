from sklearn import svm

class IrisModel():
    def __init__(self):
        self.model = svm.SVC
        self.classifier = self.model(gamma="scale")
        # do some initialization
        
    def train(self, X, y):
        self.classifier.fit(X,y)
        
