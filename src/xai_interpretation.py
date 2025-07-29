import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
import seaborn as sns

def explain_with_shap(model, X_train, X_test, save_path="results/shap_summary.png"):
    model_name = model.__class__.__name__.lower()

    if "xgb" in model_name or "catboost" in model_name or "lgbm" in model_name:
        explainer = shap.Explainer(model, X_train)
    elif "rf" in model_name or "randomforest" in model_name:
        explainer = shap.TreeExplainer(model, X_train)
    elif "gbm" in model_name or "gradientboosting" in model_name:
        explainer = shap.Explainer(model, X_train)
    elif "logistic" in model_name or "linear" in model_name:
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.Explainer(model.predict, X_train)

    if isinstance(X_test, pd.DataFrame) and len(X_test) > 100:
        X_test = X_test.sample(n=100, random_state=42)

    shap_values = explainer(X_test)

    if hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
        shap_values_to_plot = shap_values[:, :, 1]  # PTSD class
    else:
        shap_values_to_plot = shap_values

    plt.figure()
    shap.summary_plot(shap_values_to_plot, X_test, show=False)
    plt.savefig(save_path)
    plt.close()


def explain_with_lime(model, X_train, X_test, feature_names=None, instance_index=0):
    from lime import lime_tabular
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names or X_train.columns.tolist(),
        mode="classification",
        class_names=["Healthy", "PTSD"]
    )
    explanation = explainer.explain_instance(
        X_test.iloc[instance_index].values,
        model.predict_proba,
        num_features=10
    )
    explanation.show_in_notebook(show_table=True)


def plot_permutation_importance(model, X_test, y_test, feature_names):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig("results/permutation_importance.png")
    plt.close()


def plot_model_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), importances[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel("Feature Importance")
        plt.title("Model-Based Feature Importance")
        plt.tight_layout()
        plt.savefig("results/model_feature_importance.png")
        plt.close()


def plot_elasticnet_weights(X, y):
    model = ElasticNetCV(cv=5, random_state=42)
    model.fit(X, y)
    coef = model.coef_

    sorted_idx = np.argsort(np.abs(coef))[::-1]
    feature_names = np.array(X.columns)[sorted_idx]
    sorted_coef = coef[sorted_idx]
    colors = ['green' if val > 0 else 'red' for val in sorted_coef]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_names, y=sorted_coef, palette=colors)
    plt.xticks(rotation=45, ha='right')
    plt.title("ElasticNet Feature Coefficients")
    plt.tight_layout()
    plt.savefig("results/elasticnet_weights.png")
    plt.close()