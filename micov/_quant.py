import warnings

import numpy as np
import polars as pl

from ._constants import (
    COLUMN_GENOME_ID,
    COLUMN_SAMPLE_ID,
    COLUMN_START_DTYPE,
    COLUMN_STOP_DTYPE,
)

warnings.simplefilter("ignore", category=pl.exceptions.PerformanceWarning)


def make_csv_ready(df):
    return df.with_columns(
        (
            pl.lit("[")
            + pl.col(x).list.eval(pl.element().cast(str)).list.join(",")
            + pl.lit("]")
        ).alias(x)
        for x, y in df.schema.items()
        if y == pl.List(pl.String) or y == pl.List(pl.Int64)
    )


def create_bin_list(genome_length, bin_num):
    # note that bin_list is adjusted to 1-indexed to be compatible with pl.cut
    bin_list_pos_stop = (
        pl.Series("a", [0, genome_length], strict=False)
        .hist(bin_count=bin_num)
        .lazy()
        .select(pl.col("breakpoint").round().cast(COLUMN_STOP_DTYPE).alias("bin_stop"))
        .with_row_index("bin_idx", offset=1)
    )
    bin_list_pos_start = (
        pl.Series("a", [0, genome_length], strict=False)
        .hist(bin_count=bin_num)
        .lazy()
        .select(
            pl.col("breakpoint").round().cast(COLUMN_START_DTYPE).alias("bin_start")
        )
        .with_row_index("bin_idx", offset=2)
    )
    bin_list = (
        bin_list_pos_start.join(bin_list_pos_stop, on="bin_idx", how="right")
        .fill_null(0)
        .select([pl.col("bin_idx"), pl.col("bin_start"), pl.col("bin_stop")])
        .with_columns(pl.col("bin_idx").cast(COLUMN_START_DTYPE))
    )
    # setting the bin_stop of the last bin to be exactly the genome length + 1
    bin_list = bin_list.with_columns(
        pl.when(pl.col("bin_idx") == bin_num)
        .then(genome_length + 1)
        .otherwise(pl.col("bin_stop"))
        .alias("bin_stop")
    ).lazy()
    return bin_list


def pos_to_bins(pos, variable, bin_num, genome_length):
    # genome_length = pos.select("length").limit(1).collect().item()
    bin_list = create_bin_list(genome_length, bin_num)

    # get start_bin_idx and stop_bin_idx
    bin_edges = [0.0] + bin_list.select(  # noqa: RUF005
        pl.col("bin_stop")
    ).collect().to_series().to_list()
    labels = np.arange(len(bin_edges) + 1).astype(str)

    return (
        pos.with_columns(
            pl.col("start")
            .cut(bin_edges, labels=labels, left_closed=True)
            .cast(COLUMN_START_DTYPE)
            .alias("start_bin_idx"),
            pl.col("stop")
            .cut(bin_edges, labels=labels, left_closed=False)
            .cast(COLUMN_STOP_DTYPE)
            .alias("stop_bin_idx")
            + 1,
        )
        .with_columns(pl.int_ranges("start_bin_idx", "stop_bin_idx").alias("bin_idx"))
        .explode("bin_idx")
        .group_by(COLUMN_GENOME_ID, variable, "bin_idx")
        .agg(
            pl.col("start").len().alias("read_hits"),
            pl.col(COLUMN_SAMPLE_ID).n_unique().alias("sample_hits"),
            pl.col(COLUMN_SAMPLE_ID).unique().sort().alias("samples"),
        )
        .sort(by=["bin_idx", variable])
        .join(bin_list, how="left", on="bin_idx")
    )
