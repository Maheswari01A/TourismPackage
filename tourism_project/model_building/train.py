import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow

from xgboost import XGBClassifier

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. Load the prepared training and testing datasets
# ============================================================

DATA_DIR = "tourism_project/data/processed"
DEPLOYMENT_DIR = "tourism_project/deployment"

os.makedirs(DEPLOYMENT_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(DATA_DIR, "Xtrain.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "Xtest.csv"))

y_train = pd.read_csv(os.path.join(DATA_DIR, "ytrain.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, "ytest.csv")).squeeze()

y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ============================================================
# 2. Identify numerical and categorical features
# ============================================================

numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "PitchSatisfactionScore",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Passport",
    "OwnCar",
    "ProductPitched"
]

expected_features = numeric_features + categorical_features

# Validate that the prepared datasets contain the expected columns.
missing_features = [
    col for col in expected_features
    if col not in X_train.columns
]

if missing_features:
    raise ValueError(
        "Missing expected features: {}".format(missing_features)
    )

if set(X_train.columns) != set(expected_features):
    raise ValueError("Unexpected or missing columns in training data.")

if set(X_test.columns) != set(expected_features):
    raise ValueError("Training and testing columns do not match.")

# Maintain the same feature order in both datasets.
X_train = X_train[expected_features]
X_test = X_test[expected_features]


# ============================================================
# 3. Calculate the class imbalance ratio
# ============================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if positive_count == 0:
    raise ValueError("No positive samples found in training data.")

# XGBoost uses this parameter to give additional weight to
# the minority class during training.
scale_pos_weight = negative_count / positive_count

print("\nClass distribution:")
print("Negative samples:", negative_count)
print("Positive samples:", positive_count)
print("Scale positive weight:", scale_pos_weight)


# ============================================================
# 4. Define preprocessing
# ============================================================

# Scale numerical features and one-hot encode categorical features.
# Unknown categories in test/deployment data are ignored.
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
)


# ============================================================
# 5. Define XGBoost model and preprocessing pipeline
# ============================================================

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=1
)

pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


# ============================================================
# 6. Define hyperparameter search space
# ============================================================

# The search focuses on controlling model complexity and
# reducing overfitting while allowing the model to learn
# useful patterns from the minority class.

param_distributions = {

    # Number of boosting trees
    "xgbclassifier__n_estimators": [150, 250, 350],

    # Maximum tree depth
    "xgbclassifier__max_depth": [2, 3, 4],

    # Learning rate
    "xgbclassifier__learning_rate": [0.02, 0.05, 0.08],

    # Minimum weight required to create a child node
    "xgbclassifier__min_child_weight": [3, 5, 8],

    # Minimum loss reduction required for a split
    "xgbclassifier__gamma": [0, 0.5, 1],

    # Row and feature sampling to reduce overfitting
    "xgbclassifier__subsample": [0.7, 0.85, 1.0],
    "xgbclassifier__colsample_bytree": [0.7, 0.85, 1.0],

    # L1 and L2 regularization
    "xgbclassifier__reg_alpha": [0, 0.5, 1],
    "xgbclassifier__reg_lambda": [5, 10, 15],

    # Compare normal and imbalance-weighted training
    "xgbclassifier__scale_pos_weight": [1.0, scale_pos_weight]
}


# ============================================================
# 7. Configure group-aware cross-validation
# ============================================================

# Identical predictor profiles are assigned to the same group.
# This prevents matching profiles from appearing in different
# cross-validation folds.

groups = pd.factorize(
    pd.MultiIndex.from_frame(X_train)
)[0]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 8. Configure MLflow experiment tracking
# ============================================================

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("Visit_with_Us_Model_Training")


# ============================================================
# 9. Perform randomized hyperparameter search
# ============================================================

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    scoring="average_precision",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True,
    return_train_score=False
)


with mlflow.start_run(run_name="XGBoost_Randomized_Search"):

    # Log experiment configuration.
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("search_method", "RandomizedSearchCV")
    mlflow.log_param("iterations", 50)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("cv_strategy", "StratifiedGroupKFold")
    mlflow.log_param("cv_scoring", "average_precision")
    mlflow.log_param("feature_count", len(expected_features))

    # Fit only on the training dataset.
    search.fit(
        X_train,
        y_train,
        groups=groups
    )

    # RandomizedSearchCV refits the best estimator on all training data.
    best_model = search.best_estimator_

    print("\nBest parameters:")
    print(search.best_params_)

    print("\nBest CV Average Precision:")
    print(search.best_score_)

    mlflow.log_params(search.best_params_)
    mlflow.log_metric(
        "best_cv_average_precision",
        float(search.best_score_)
    )


    # ========================================================
    # 10. Log individual hyperparameter search trials
    # ========================================================

    cv_results = search.cv_results_

    for i in range(len(cv_results["params"])):

        with mlflow.start_run(
            run_name="Trial_{}".format(i + 1),
            nested=True
        ):

            mlflow.log_params(cv_results["params"][i])

            mlflow.log_metric(
                "mean_cv_average_precision",
                float(cv_results["mean_test_score"][i])
            )

            mlflow.log_metric(
                "std_cv_average_precision",
                float(cv_results["std_test_score"][i])
            )

            mlflow.log_metric(
                "cv_rank",
                int(cv_results["rank_test_score"][i])
            )


    # ========================================================
    # 11. Select classification threshold using OOF predictions
    # ========================================================

    # Generate out-of-fold probabilities using training data only.
    # The test dataset is not used for threshold selection.

    oof_probabilities = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=cv,
        groups=groups,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    # Evaluate F1 across candidate thresholds.
    thresholds = np.arange(0.20, 0.81, 0.01)

    threshold_results = []

    for threshold in thresholds:

        oof_predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_train,
            oof_predictions,
            zero_division=0
        )

        threshold_results.append((threshold, score))

    # Select the threshold producing the highest OOF F1.
    best_threshold, best_oof_f1 = max(
        threshold_results,
        key=lambda item: item[1]
    )

    print("\nSelected threshold:", best_threshold)
    print("OOF F1 score:", best_oof_f1)

    mlflow.log_param(
        "classification_threshold",
        float(best_threshold)
    )

    mlflow.log_metric(
        "oof_f1_at_selected_threshold",
        float(best_oof_f1)
    )


    # ========================================================
    # 12. Evaluate model on training and testing datasets
    # ========================================================

    # Use the same selected threshold for both datasets so that
    # training and testing metrics can be compared fairly.

    train_probabilities = best_model.predict_proba(
        X_train
    )[:, 1]

    train_predictions = (
        train_probabilities >= best_threshold
    ).astype(int)

    test_probabilities = best_model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)


    # --------------------------------------------------------
    # Training evaluation
    # --------------------------------------------------------

    print("\nTRAIN CLASSIFICATION REPORT")

    print(
        classification_report(
            y_train,
            train_predictions,
            zero_division=0
        )
    )

    print("\nTRAIN CONFUSION MATRIX")
    print(confusion_matrix(y_train, train_predictions))

    train_accuracy = accuracy_score(y_train, train_predictions)
    train_precision = precision_score(
        y_train, train_predictions, zero_division=0
    )
    train_recall = recall_score(
        y_train, train_predictions, zero_division=0
    )
    train_f1 = f1_score(
        y_train, train_predictions, zero_division=0
    )
    train_roc_auc = roc_auc_score(y_train, train_probabilities)
    train_average_precision = average_precision_score(
        y_train, train_probabilities
    )

    print("\nTRAIN METRICS")
    print("Accuracy:", train_accuracy)
    print("Precision:", train_precision)
    print("Recall:", train_recall)
    print("F1:", train_f1)
    print("ROC AUC:", train_roc_auc)
    print("Average Precision:", train_average_precision)


    # --------------------------------------------------------
    # Testing evaluation
    # --------------------------------------------------------

    print("\nTEST CLASSIFICATION REPORT")

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    print("\nTEST CONFUSION MATRIX")
    print(confusion_matrix(y_test, test_predictions))

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    test_recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    test_f1 = f1_score(
        y_test, test_predictions, zero_division=0
    )
    test_roc_auc = roc_auc_score(y_test, test_probabilities)
    test_average_precision = average_precision_score(
        y_test, test_probabilities
    )

    print("\nTEST METRICS")
    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1:", test_f1)
    print("ROC AUC:", test_roc_auc)
    print("Average Precision:", test_average_precision)

    # Log the existing test metrics to MLflow.
    mlflow.log_metric("test_accuracy", float(test_accuracy))
    mlflow.log_metric("test_precision", float(test_precision))
    mlflow.log_metric("test_recall", float(test_recall))
    mlflow.log_metric("test_f1", float(test_f1))
    mlflow.log_metric("test_roc_auc", float(test_roc_auc))
    mlflow.log_metric(
        "test_average_precision",
        float(test_average_precision)
    )


    # ========================================================
    # 13. Save the trained model and decision threshold
    # ========================================================

    model_path = os.path.join(
        DEPLOYMENT_DIR,
        "tourism_xgb_model.joblib"
    )

    threshold_path = os.path.join(
        DEPLOYMENT_DIR,
        "decision_threshold.json"
    )

    # Save the complete pipeline, including preprocessing.
    joblib.dump(best_model, model_path)

    # Save the selected classification threshold for deployment.
    with open(threshold_path, "w") as file:
        json.dump(
            {"threshold": float(best_threshold)},
            file,
            indent=4
        )

    # Store the saved files as MLflow artifacts.
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(threshold_path)

    print("\nModel saved:", model_path)
    print("Threshold saved:", threshold_path)


print("\nModel training completed successfully.")
