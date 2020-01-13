from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class SmoteEnsemble:
    
    def __init__(self):
        #
        self.smote = SMOTE()
        self.models = [
            ("Ada",AdaBoostClassifier()),
            ("Bagging",BaggingClassifier(base_estimator=DecisionTreeClassifier())),
            ("RandomForest",RandomForestClassifier())
        ]
        self.model=None
    
    def fit(self,X,y,test_size=0.2):
        if self.model is None:
	
            #Determine best performing model
            best = ""
            best_f1 = -1
            #
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            #
            for name, model in self.models:
                model.fit(X_train,y_train)
                #
                y_pred = model.predict(X_test)
                #
                f1 = f1_score(y_test,y_pred)
                #
                if f1>best_f1:
                    best_f1 = f1
                    best = name
                    
            #SMOTE Train data
            X, y = self.smote.fit_resample(X,y)
            
            #Train the best model
            model = None
            if best=="Ada":
                model = AdaBoostClassifier()
            elif best=="Bagging":
                model = BaggingClassifier(base_estimator=DecisionTreeClassifier())
            elif best=="RandomForest":
                model = RandomForestClassifier()
            #
            model.fit(X,y)
            
            #Store trained model
            self.model = model
        else:
            self.model.fit(X,y)
    
    def predict(self,X):
        return self.model.predict(X)
