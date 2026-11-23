from pathlib import Path

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
import os


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

from sklearn.cluster import AffinityPropagation


def group_features_by_correlation(df):
    feature_df = df.iloc[:, 3:].copy()
    feature_names = feature_df.columns.tolist()

    corr = feature_df.corr(method='pearson').abs()
    similarity = corr.values

    off_diag = similarity[~np.eye(similarity.shape[0], dtype=bool)]
    preference = np.median(off_diag)

    ap = AffinityPropagation(
        affinity='precomputed',
        preference=preference,
        random_state=42,
        max_iter=500,
        convergence_iter=30
    )
    ap.fit(similarity)

    labels = ap.labels_

    cluster_dict = {}
    for feat, label in zip(feature_names, labels):
        cluster_dict.setdefault(label, []).append(feat)

    groups = []
    independent = []

    for cluster_features in cluster_dict.values():
        if len(cluster_features) == 1:
            independent.append(cluster_features[0])
        else:
            groups.append(cluster_features)

    return independent, groups







def select_top_n_features_by_rmse(groups, df_features, df_target, target_column, model_use, top_n=3):
    result_groups = []
    for group in groups:
        if len(group) <= top_n:
            result_groups.append(group)
            continue

        rmse_list = []
        for feat in group:
            X = df_features[[feat]]
            y = df_target[target_column]

            model = model_use()
            model.fit(X, y)
            y_pred = model.predict(X)

            rmse = root_mean_squared_error(y, y_pred)
            rmse_list.append((feat, rmse))

        top_feats = [f for f, _ in sorted(rmse_list, key=lambda x: x[1])[:top_n]]
        result_groups.append(top_feats)

    return result_groups


def evaluate_combination(comb, df_features, y, model_class):
    X = df_features[list(comb)].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = model_class()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    return comb, rmse



def select_best_across_groups(groups, df_features, df_target, target_column, model_class, n_jobs=-1):
    y = df_target[target_column].values
    combinations = list(itertools.product(*groups))
    print(f"Evaluating {len(combinations)} combinations...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_combination)(comb, df_features, y, model_class)
        for comb in combinations
    )
    best_comb, best_rmse = min(results, key=lambda x: x[1])
    return list(best_comb), best_rmse


def select_features_by_importance(mdi, pi_means, method='combined', top_n=13, alpha=0.5):
    pi_aligned = pi_means.reindex(mdi.index).fillna(0)
    if method == 'mdi':
        ranked = mdi.sort_values(ascending=False)
    elif method == 'pi':
        ranked = pi_aligned.sort_values(ascending=False)
    elif method == 'combined':
        mdi_norm = (mdi - mdi.min()) / (mdi.max() - mdi.min())
        pi_norm = (pi_aligned - pi_aligned.min()) / (pi_aligned.max() - pi_aligned.min())
        combined_score = alpha * mdi_norm + (1 - alpha) * pi_norm
        ranked = combined_score.sort_values(ascending=False)
    else:
        raise ValueError("method must be mdi, pi, or combined")
    return ranked.index[:top_n].tolist()


def features_by_importance(df, target_col, independent, best_feats, model_use, sheet_name, model_short_name, top_n=13):
    os.makedirs("results/plots", exist_ok=True)

    y = df[target_col]
    last_groups = independent + best_feats
    X = df[last_groups]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = model_use()
    model.fit(X_train, y_train)

    mdi = pd.Series(model.feature_importances_, index=X.columns)
    pi_result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)
    pi_means = pd.Series(pi_result.importances_mean, index=X.columns)
    pi_values = pi_result.importances

    mdi_sorted = mdi.sort_values(ascending=False)
    pi_means_sorted = pi_means.sort_values(ascending=False)
    pi_sorted_values = pi_values[[X.columns.get_loc(feat) for feat in pi_means_sorted.index]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=False)
    axes[0].barh(range(len(mdi_sorted)), mdi_sorted.values, color='steelblue')
    axes[0].set_yticks(range(len(mdi_sorted)))
    axes[0].set_yticklabels(mdi_sorted.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('MDI Importance')
    axes[0].set_title('Feature Importance (MDI) via ExtraTrees')

    axes[1].boxplot(pi_sorted_values.T, vert=False, labels=pi_means_sorted.index)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Permutation Importance')
    axes[1].set_title('Permutation Importance via ExtraTrees')

    plt.tight_layout()
    plot_name = f"results/plots/{model_short_name}_{sheet_name}_MDI_vs_PI_boxplots.png"
    plt.savefig(plot_name, dpi=300)
    plt.close()

    return select_features_by_importance(mdi, pi_means, 'combined', top_n)


def kfold_rmse(X, y, model_use, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = model_use()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        errors.extend((y_test - y_pred) ** 2)
    return np.sqrt(np.mean(errors))


def evaluate_subset(subset, X, y, model_use, k):
    rmse = kfold_rmse(X[list(subset)], y, model_use, k)
    return subset, rmse


def Feature_Subset_Exhaustive_Search(df, target_col, selected_features, model_use, model_short_name, sheet_name, k_0=10):
    os.makedirs("results/plots", exist_ok=True)
    y = df[target_col]
    X = df[selected_features]
    N = len(selected_features)
    k = k_0

    rmse_all = []
    rmse_min_curve = []
    best_feature_sets = []

    for M in range(1, N + 1):
        combs = list(itertools.combinations(selected_features, M))
        print(f"Evaluating {len(combs)} combinations with M={M}...")
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_subset)(subset, X, y, model_use, k) for subset in combs
        )
        for subset, rmse in results:
            rmse_all.append((len(subset), rmse))
        best_subset, min_rmse = min(results, key=lambda x: x[1])
        rmse_min_curve.append((M, min_rmse))
        best_feature_sets.append((M, best_subset))
        print(f"M={M}, minimum RMSE={min_rmse:.4f}, best subset={best_subset}")

    x_all, y_all = zip(*rmse_all)
    x_min, y_min = zip(*rmse_min_curve)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_all, y_all, alpha=0.4, label='All Subset RMSEs', color='gray')
    plt.plot(x_min, y_min, marker='o', linestyle='-', color='red', label='Minimum RMSE per M')
    plt.xlabel('Number of Features (M)')
    plt.ylabel(f'{k}-Fold CV RMSE')
    plt.title(f'Feature Subset Exhaustive Search (K={k})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_name = f"results/plots/{model_short_name}_{sheet_name}_Feature_Subset_RMSE_Curve.png"
    plt.savefig(plot_name, dpi=300)
    plt.close()

    return best_feature_sets


def run_knn_feature_pipeline(sheet_name, model_class, model_short_name):
    df = pd.read_excel(
        Path(__file__).resolve().parents[1] / 'data' / 'train.xlsx',
        sheet_name=sheet_name
    )
    target_col = 'd33'

    independent, groups = group_features_by_correlation(df)

    print("Single-feature groups:", end=" ")
    print(independent)
    for idx, g in enumerate(groups):
        print(f"Group {idx+1}:", end=" ")
        print(g)


    df_target = df[[target_col]]
    df_features = df.drop(columns=[target_col])

    top_features_per_group = select_top_n_features_by_rmse(
        groups, df_features, df_target, target_col, model_class, top_n=3
    )
    for i, group in enumerate(top_features_per_group, 1):
        print(f"Group {i} after screening: {group}")

    best_feats, best_rmse = select_best_across_groups(
        top_features_per_group, df_features, df_target, target_col, model_class
    )
    print(f"Best combination: {best_feats}")
    print(f"Minimum RMSE: {best_rmse:.4f}")

    selected_features = features_by_importance(
        df, target_col, independent, best_feats, model_class,
        sheet_name=sheet_name, model_short_name=model_short_name, top_n=11
    )
    print(f"Features retained by importance: {selected_features}")

    best_feature_sets = Feature_Subset_Exhaustive_Search(
        df, target_col, selected_features, model_class,
        model_short_name=model_short_name, sheet_name=sheet_name
    )
    print(best_feature_sets)

if __name__ == "__main__":
    model_classes = [GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor]
    model_short_names = ["GBR", "ETR", "RFR"]
    for model_class, short_name in zip(model_classes, model_short_names):
        print(f"\nRunning feature selection for {short_name}")
        run_knn_feature_pipeline("Train", model_class, short_name)
