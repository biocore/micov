import glob

import numpy as np
import pandas as pd
from scipy.stats import entropy


def gini(array):
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def compute_entropy(group):
    p = group.value_counts(normalize=True)
    return entropy(p, base=2)


def rank_sample_groups(group_df):
    # entropy lower, distribution more variable
    entropy_rank = group_df["entropy"].rank(ascending=True)
    # gini higher, distribution more variable
    gini_rank = group_df["gini"].rank(ascending=False)
    # CV higher, distribution more variable
    cv_rank = group_df["cv"].rank(ascending=False)

    group_df["ranking"] = (entropy_rank + gini_rank + cv_rank).rank(
        method="dense", ascending=True
    )  # tie wil be given the same rank

    return group_df


def choose_most_variable_group(group_df):
    group_df = rank_sample_groups(group_df)
    best_group = group_df["ranking"].idxmin()
    return group_df.loc[[best_group]]


def rank_genome_of_interest(plotdir):
    all_files = glob.glob(f"{plotdir}/*.tsv.gz")
    all_files = [file for file in all_files if ".ks.tsv" not in file]
    metrics_selected = []

    for file in all_files:
        df = pd.read_csv(file, sep="\t", compression="gzip")

        # Compute variability metrics for each group in this file
        y_distribution = df.groupby(["group", "y"])["x"].count().reset_index()
        y_distribution.rename(columns={"x": "count_x"}, inplace=True)

        # Compute all metrics per group
        metrics = (
            y_distribution.groupby("group")["count_x"]
            .agg(mean="mean", std="std", entropy=compute_entropy, gini=gini)
            .assign(cv=lambda x: x["std"] / x["mean"])
        )

        selected_group = choose_most_variable_group(metrics)
        selected_group["filename"] = file.split("/")[-1]
        metrics_selected.append(selected_group)

    metrics_df = pd.concat(metrics_selected)

    genomes_ranked = rank_sample_groups(metrics_df).sort_values("ranking")

    genomes_ranked = genomes_ranked.reset_index().rename(columns={"index": "group"})
    column_order = ["filename", "group"] + [
        col for col in genomes_ranked.columns if col not in ["filename", "group"]
    ]
    genomes_ranked = genomes_ranked[column_order]  # reorder columns

    return genomes_ranked
