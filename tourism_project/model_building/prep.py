
import os
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --------------------------------------------------
# 1. Load the raw dataset
# --------------------------------------------------

RAW_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(RAW_PATH)

print("Raw dataset loaded successfully.")
print("Original shape:", df.shape)


# --------------------------------------------------
# 2. Validate expected columns
# --------------------------------------------------

expected_columns = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

missing_columns = [
    col for col in expected_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing expected columns: {missing_columns}"
    )

if df["ProdTaken"].isnull().any():
    raise ValueError("Target column contains missing values.")

if not set(df["ProdTaken"].unique()).issubset({0, 1}):
    raise ValueError("ProdTaken must contain only 0 and 1.")


# --------------------------------------------------
# 3. Remove unnecessary columns
# --------------------------------------------------

# CustomerID is an identifier.
# Unnamed: 0 is an index-like column.

df.drop(
    columns=["CustomerID", "Unnamed: 0"],
    errors="ignore",
    inplace=True
)

print("Shape after removing unnecessary columns:", df.shape)


# --------------------------------------------------
# 4. Handle missing values
# --------------------------------------------------

missing_values = df.isnull().sum()

if missing_values.sum() > 0:
    raise ValueError(
        "Missing values found:\n"
        + str(missing_values[missing_values > 0])
    )

print("Missing value check passed.")


# --------------------------------------------------
# 5. Standardize categorical values
# --------------------------------------------------

df["Gender"] = df["Gender"].replace({
    "Fe Male": "Female"
})

df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Unmarried": "Single"
})

print("Categorical values standardized.")


# --------------------------------------------------
# 6. Check negative values
# --------------------------------------------------

numeric_columns = df.select_dtypes(include="number").columns

negative_counts = (df[numeric_columns] < 0).sum()
negative_counts = negative_counts[negative_counts > 0]

if not negative_counts.empty:
    raise ValueError(
        "Negative values found:\n" + str(negative_counts)
    )

print("Negative value check passed.")


# --------------------------------------------------
# 7. Define features and target
# --------------------------------------------------

# Prediction point for this experiment:
# After the sales pitch, before the final purchase decision.
#
# Therefore, interaction-related features are included.
#
# ProductPitched and Designation have a one-to-one mapping.
# Keep ProductPitched and exclude Designation.

excluded_columns = [
    "Designation"
]

X = df.drop(columns=["ProdTaken"] + excluded_columns)
y = df["ProdTaken"]

print("\nPredictor columns:")
print(list(X.columns))


# --------------------------------------------------
# 8. Identify duplicate predictor profiles
# --------------------------------------------------

# Identical predictor profiles are assigned to the same
# group to prevent train/test leakage.
#
# Records with identical predictors but different target
# values are retained and reported, not discarded.

groups = pd.factorize(
    pd.MultiIndex.from_frame(X)
)[0]

group_target_counts = (
    pd.DataFrame({
        "group": groups,
        "target": y.values
    })
    .groupby("group")["target"]
    .nunique()
)

mixed_target_groups = group_target_counts[
    group_target_counts > 1
]

print("\nDuplicate profile check:")
print("Total records:", len(X))
print("Unique predictor profiles:", len(set(groups)))
print(
    "Repeated predictor profiles:",
    pd.Series(groups).value_counts().gt(1).sum()
)
print(
    "Profiles with conflicting target values:",
    len(mixed_target_groups)
)

if len(mixed_target_groups) > 0:
    print(
        "Note: Conflicting target profiles are retained "
        "and kept within the same split."
    )


# --------------------------------------------------
# 9. Stratified group-aware train/test split
# --------------------------------------------------

# Five folds provide an approximately 80:20 split.
# Identical predictor profiles stay in the same fold.

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

best_split = None
best_score = float("inf")

overall_positive_rate = y.mean()

for train_idx, test_idx in sgkf.split(X, y, groups):

    actual_test_size = len(test_idx) / len(X)
    test_positive_rate = y.iloc[test_idx].mean()

    score = (
        abs(actual_test_size - 0.20)
        + abs(test_positive_rate - overall_positive_rate)
    )

    if score < best_score:
        best_score = score
        best_split = (train_idx, test_idx)

if best_split is None:
    raise ValueError("Unable to create train/test split.")

train_idx, test_idx = best_split

Xtrain = X.iloc[train_idx].copy()
Xtest = X.iloc[test_idx].copy()

ytrain = y.iloc[train_idx].copy()
ytest = y.iloc[test_idx].copy()


# --------------------------------------------------
# 10. Validate train/test split
# --------------------------------------------------

train_groups = set(groups[train_idx])
test_groups = set(groups[test_idx])

overlapping_groups = train_groups.intersection(test_groups)

if overlapping_groups:
    raise ValueError(
        "Data leakage detected: identical predictor profiles "
        "exist in both training and testing datasets."
    )

print("\nTrain/test split completed successfully.")

print("Training records:", len(Xtrain))
print("Testing records:", len(Xtest))

print("\nTraining target distribution (%):")
print(ytrain.value_counts(normalize=True) * 100)

print("\nTesting target distribution (%):")
print(ytest.value_counts(normalize=True) * 100)

print("\nOverlapping duplicate groups:", len(overlapping_groups))


# --------------------------------------------------
# 11. Save training and testing datasets
# --------------------------------------------------

OUTPUT_PATH = "tourism_project/data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

Xtrain.to_csv(
    os.path.join(OUTPUT_PATH, "Xtrain.csv"),
    index=False
)

Xtest.to_csv(
    os.path.join(OUTPUT_PATH, "Xtest.csv"),
    index=False
)

ytrain.to_csv(
    os.path.join(OUTPUT_PATH, "ytrain.csv"),
    index=False
)

ytest.to_csv(
    os.path.join(OUTPUT_PATH, "ytest.csv"),
    index=False
)

print("\nData preparation completed.")
print("Training and testing files saved successfully.")
