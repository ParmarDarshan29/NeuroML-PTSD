import shap
import lime.lime_tabular
import eli5
from eli5.sklearn import PermutationImportance


def explain_with_shap(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    return shap_values

def explain_with_lime(model, X, selected_features, index=3):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X[selected_features].values,
        feature_names=selected_features,
        class_names=['Healthy control', 'Posttraumatic stress disorder'],
        mode='classification'
    )
    exp = explainer.explain_instance(
        data_row=X.iloc[index][selected_features].values,
        predict_fn=model.predict_proba
    )
    return exp

def explain_with_eli5(model, X, selected_features):
    perm = PermutationImportance(model, random_state=42).fit(X[selected_features], y)
    return eli5.show_weights(perm, feature_names=selected_features.tolist())