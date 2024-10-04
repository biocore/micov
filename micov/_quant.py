import numpy as np
import polars as pl


# sherlyn: from input to bin_df for a single genomem cli.py
# where is the pos coming from (need to build a schema
# [genome_id, start, stop, sample_id])
def pos_to_bins(pos, genome_length, bin_num):
    obs_count, bin_edges = np.histogram(
        pos.select(pl.col("start")), bins=bin_num, range=(0, genome_length)
    )
    bin_list = pl.DataFrame(
        {
            "bin_idx": np.arange(len(bin_edges) - 1),
            "bin_start": bin_edges[:-1].astype(int),  # Start of each bin
            "bin_stop": bin_edges[1:].astype(int),  # End of each bin
        }
    )

    start_bin_idx = pl.Series(
        "start_bin_idx",
        np.digitize(pos.select(pl.col("start")).to_series(), bins=bin_edges),
    )
    stop_bin_idx = pl.Series(
        "stop_bin_idx",
        np.digitize(pos.select(pl.col("stop")).to_series(), bins=bin_edges),
    )

    pos = pos.with_columns(
        [start_bin_idx - 1, stop_bin_idx - 1]  # Adjust for 0-indexing
    )

    # Adjust stop_bin_idx if stop equals bin_start
    pos = (
        pos.join(
            bin_list, how="left", left_on="stop_bin_idx", right_on="bin_idx"
        )
        .with_columns(
            pl.when(pl.col("stop") == pl.col("bin_start"))
            .then(pl.col("stop_bin_idx") - 1)
            .otherwise(pl.col("stop_bin_idx"))
            .alias("stop_bin_idx")
        )
        .drop(["bin_start", "bin_stop"])
    )

    # Update stop_bin_idx +1 for pl.arange and generate range of bins
    pos = pos.with_columns(
        (pl.col("stop_bin_idx") + 1).alias("stop_bin_idx_add1")
    )

    # Generate the range of bins covered
    pos = pos.with_columns(
        pl.int_ranges("start_bin_idx", "stop_bin_idx_add1").alias("bin_idx")
    ).drop("stop_bin_idx_add1")

    # Generate bin_df
    bin_df = (
        pos.explode("bin_idx")
        .group_by("bin_idx")
        .agg(
            pl.col("start").len().alias("read_hits"),
            pl.col("sample_id").n_unique().alias("sample_hits"),
            pl.col("sample_id").unique().alias("samples"),
        )
        .sort(by="bin_idx")
        .join(bin_list, how="left", on="bin_idx")
    )

    return bin_df, pos


# caitlin: micov_calc, micov_main, _plots
# - a polar dataframe of bin_df + other stuff
# - pl.DataFrame(bin_df)

# unit tests

if __name__ == "__main__":
    input1 = ""
    input2 = ""

    data = {
        "genome_id": ["G000006605", "G000006605", "G000006605", "G000006605"],
        "start": [5, 2, 11, 54],
        "stop": [7, 7, 15, 59],
        "sample_id": ["A", "B", "A", "C"],
    }
    df = pl.DataFrame(data)
    genome_length = 100
    bin_size = 10

    # run function
    res1, res2 = pos_to_bins(df, genome_length, bin_size)
    print(res1)
    print(res2)
