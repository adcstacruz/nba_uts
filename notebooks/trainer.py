import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


random_state = 42 

# ------- DEFINE PARAMETER GRID -------
# TODO: Make a config file if needed
rf_param_grid = {
    'n_estimators': [1000],
    'max_depth': [None, 10, 20],
    # 'min_samples_split': [2, 5, 10],
    }

et_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    }

nn_param_grid = {
    'hidden_layer_sizes': [(50, 50), (100, 100), (50,)],
    'activation': ['relu', 'tanh'],
    }    

param_grids = {
    'rf': rf_param_grid,
    'et': et_param_grid,
    'nn': nn_param_grid, 
}

classifier_dict = {
    'rf': RandomForestClassifier(),
    'et': ExtraTreesClassifier(),
    'nn': MLPClassifier(),
}

# ------- MODEL TRAINER -------
class ModelTrainer:
    def __init__(self, model_name, param_grid=None, cv=5):
        # Set model name
        self.model_name = model_name
        self.cv = cv
        
        # Get the search param grid
        if not param_grid:
            self.param_grid = param_grids 

        # Create model
        self.model = self._create_model()
        
    def _create_model(self):
        classifier = classifier_dict[self.model_name]
        if self.model_name in ['rf', 'et']:
            pipeline = Pipeline([
                ('classifier', classifier),
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', classifier),
            ])
        return pipeline
    
    def train(self, X, y):
        # with mlflow.start_run():
        #     mlflow.log_params(self.param_grid)
        self.model.fit(X, y)
            
    def evaluate(self, X,y):
        y_preds = cross_val_predict(self.model, X, y, cv=self.cv, method='predict_proba')
        y_pred = y_preds.argmax(axis=1)
    
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_preds[:, 1])
        
        # mlflow.log_metrics({
        #     'accuracy': accuracy,
        #     'precision': precision,
        #     'recall': recall,
        #     'f1_score': f1,
        #     'roc_auc_score': roc_auc,
        # })
        return accuracy, precision, recall, f1, roc_auc
