import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet

def explain_with_shap(model, X_train, X_test, plot_type="summary", max_display=20):
    """SHAP explanation: summary or bar plot"""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    if plot_type == "summary":
        shap.plots.beeswarm(shap_values, max_display=max_display)
    elif plot_type == "bar":
        shap.plots.bar(shap_values, max_display=max_display)
    else:
        raise ValueError("Choose plot_type from ['summary', 'bar']")

def explain_with_lime(model, X_train, X_test, feature_names=None, sample_index=0, class_names=["Healthy", "PTSD"]):
    """LIME explanation on a single sample"""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names if feature_names is not None else X_train.columns,
        class_names=class_names,
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        data_row=X_test.iloc[sample_index],
        predict_fn=model.predict_proba
    )
    exp.show_in_notebook()

def plot_permutation_importance(model, X_test, y_test, feature_names=None):
    """Sklearn's Permutation Feature Importance"""
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc')
    
    sorted_idx = result.importances_mean.argsort()[::-1]
    top_features = np.array(feature_names if feature_names is not None else X_test.columns)[sorted_idx][:20]
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], result.importances_mean[sorted_idx][:20][::-1])
    plt.xlabel("Permutation Importance (mean decrease in ROC AUC)")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

def plot_model_feature_importance(model, feature_names=None, top_n=20):
    """For models that support .feature_importances_"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:top_n]
        names = np.array(feature_names if feature_names is not None else range(len(importances)))
        
        plt.figure(figsize=(10, 6))
        plt.barh(names[sorted_idx][::-1], importances[sorted_idx][::-1])
        plt.xlabel("Feature Importance")
        plt.title("Model-based Feature Importances")
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not support feature_importances_ attribute.")

def plot_elasticnet_weights(X, y, top_k=20):
    """Display top features selected by ElasticNet"""
    model = ElasticNet(random_state=42)
    model.fit(X, y)
    coefs = pd.Series(np.abs(model.coef_), index=X.columns)
    top_features = coefs.sort_values(ascending=False).head(top_k)
    
    plt.figure(figsize=(10, 6))
    top_features.plot(kind='barh')
    plt.xlabel("ElasticNet Coefficient Magnitude")
    plt.title("Top Features from ElasticNet")
    plt.tight_layout()
    plt.show()
