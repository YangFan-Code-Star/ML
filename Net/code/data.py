import os

import pandas as pd
import torch

from config import DATA_ROOT, RANDOM_SEED


HOUSE_PRICE_DIR = os.path.join(DATA_ROOT, "HousePrice")
TRAIN_CSV = os.path.join(HOUSE_PRICE_DIR, "train.csv")
TEST_CSV = os.path.join(HOUSE_PRICE_DIR, "test.csv")


def _split_indices(num_samples, num_folds, fold_idx, random_seed):
    if num_samples <= 1:
        return list(range(num_samples)), []

    num_folds = max(2, min(num_folds, num_samples))
    if not 0 <= fold_idx < num_folds:
        raise ValueError(f"fold_idx must be in [0, {num_folds - 1}], got {fold_idx}")

    generator = torch.Generator().manual_seed(random_seed)
    indices = torch.randperm(num_samples, generator=generator)
    folds = [fold.tolist() for fold in torch.tensor_split(indices, num_folds)]

    val_indices = folds[fold_idx]
    train_indices = [
        index
        for current_fold, fold_indices in enumerate(folds)
        if current_fold != fold_idx
        for index in fold_indices
    ]
    return train_indices, val_indices


def _build_feature_tensors(train_df, val_df, test_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    numeric_columns = train_df.select_dtypes(include=["number"]).columns
    numeric_means = train_df[numeric_columns].mean()
    numeric_stds = train_df[numeric_columns].std().replace(0, 1).fillna(1)

    processed_frames = []
    for features_df in (train_df, val_df, test_df):
        scaled_numeric = (
            features_df[numeric_columns].fillna(numeric_means) - numeric_means
        ) / numeric_stds
        processed_df = features_df.copy()
        processed_df[numeric_columns] = scaled_numeric.astype("float32")
        processed_frames.append(processed_df)

    train_df, val_df, test_df = processed_frames
    all_features_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    all_features_df = pd.get_dummies(all_features_df, dummy_na=True).astype("float32")

    train_size = len(train_df)
    val_size = len(val_df)
    train_end = train_size
    val_end = train_size + val_size

    train_features = torch.tensor(
        all_features_df.iloc[:train_end].to_numpy(),
        dtype=torch.float32,
    )
    val_features = torch.tensor(
        all_features_df.iloc[train_end:val_end].to_numpy(),
        dtype=torch.float32,
    )
    test_features = torch.tensor(
        all_features_df.iloc[val_end:].to_numpy(),
        dtype=torch.float32,
    )
    return train_features, val_features, test_features


def preprocess_data(
    train_path=TRAIN_CSV,
    test_path=TEST_CSV,
    num_folds=5,
    fold_idx=0,
    random_seed=RANDOM_SEED,
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    test_ids = test_df["Id"].to_numpy()
    targets = torch.tensor(
        train_df["SalePrice"].to_numpy(dtype="float32"),
        dtype=torch.float32,
    ).reshape(-1, 1)
    features_df = train_df.drop(columns=["SalePrice"])
    train_indices, val_indices = _split_indices(
        len(features_df),
        num_folds,
        fold_idx,
        random_seed,
    )

    train_features_df = features_df.iloc[train_indices]
    val_features_df = features_df.iloc[val_indices]
    train_targets = targets[train_indices]
    val_targets = targets[val_indices]

    train_features, val_features, test_features = _build_feature_tensors(
        train_features_df,
        val_features_df,
        test_df,
    )
    input_dim = train_features.size(1)

    return (
        train_features,
        train_targets,
        val_features,
        val_targets,
        test_features,
        input_dim,
        test_ids,
    )
