import numpy as np
import polars as pl


# sherlyn: from input to bin_df for a single genomem cli.py
def create_bin_list(genome_length, bin_num):
    # note that bin_list is adjusted to 1-indexed to be compatible with pl.cut
    bin_list_pos_stop = (
        pl.Series("a", [0, genome_length], strict=False)
        .hist(bin_count=bin_num)
        .lazy()
        .select(pl.col("breakpoint").alias("bin_stop"))
        .with_row_index("bin_idx", offset=1)
    )
    bin_list_pos_start = (
        pl.Series("a", [0, genome_length], strict=False)
        .hist(bin_count=bin_num)
        .lazy()
        .select(pl.col("breakpoint").alias("bin_start"))
        .with_row_index("bin_idx", offset=2)
    )
    bin_list = (
        bin_list_pos_start.join(bin_list_pos_stop, on="bin_idx", how="right")
        .fill_null(0)
        .select([pl.col("bin_idx"), pl.col("bin_start"), pl.col("bin_stop")])
        .with_columns(pl.col("bin_idx").cast(pl.Int64))
        .collect()
    )
    return bin_list


def pos_to_bins(pos, genome_length, bin_num):
    pos = pos.lazy()
    bin_list = create_bin_list(genome_length, bin_num).lazy()

    # get start_bin_idx and stop_bin_idx
    bin_edges = [0.0] + bin_list.select(
        pl.col("bin_stop")
    ).collect().to_series().to_list()
    cut_start = (
        pos.select(pl.col("start"))
        .collect()
        .to_series()
        .cut(
            bin_edges,
            labels=np.arange(len(bin_edges) + 1).astype(str),
            left_closed=True,
        )
        .cast(pl.Int64)
        .alias("start_bin_idx")
    )
    cut_stop = (
        pos.select(pl.col("stop"))
        .collect()
        .to_series()
        .cut(
            bin_edges,
            labels=np.arange(len(bin_edges) + 1).astype(str),
            left_closed=False,
        )
        .cast(pl.Int64)
        .alias("stop_bin_idx")
    )
    pos = pos.with_columns([cut_start, cut_stop])

    # Update stop_bin_idx +1 for pl.arange and generate range of bins
    pos = pos.with_columns(
        (pl.col("stop_bin_idx") + 1).alias("stop_bin_idx_add1")
    )

    # Generate range of bins covered
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
            pl.col("sample_id").unique().sort().alias("samples"),
        )
        .sort(by="bin_idx")
        .join(bin_list, how="left", on="bin_idx")
    )

    return bin_df.collect(), pos.collect()


# caitlin: micov_calc, micov_main, _plots
# - a polar dataframe of bin_df + other stuff
# - pl.DataFrame(bin_df)

# unit tests

if __name__ == "__main__":
    input1 = ""
    input2 = ""

    df = pl.DataFrame(
        [
            ["G000006605", 5, 20, "A"],
            ["G000006605", 10, 17, "A"],
            ["G000006605", 11, 15, "A"],
            ["G000006605", 54, 59, "A"],
            ["G000006605", 71, 76, "B"],
            ["G000006605", 95, 99, "B"],
        ],
        orient="row",
        schema=[
            ("genome_id", str),
            ("start", int),
            ("stop", int),
            ("sample_id", str),
        ],
    )
    genome_length = 100
    bin_num = 10

    # run function
    bin_df, pos_updated = pos_to_bins(df, genome_length, bin_num)
