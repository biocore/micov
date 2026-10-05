"""Depth, breadth and per-ORF statistics for `micov depth-plot`."""

from ._constants import COLUMN_GENOME_ID, COLUMN_LENGTH, COLUMN_SAMPLE_ID
from ._io import (
    ALIGNED_ROWS,
    DEPTH_FEATURES_TABLE,
    ORFS_TABLE,
    SAMPLE_GROUPS_TABLE,
    _examples,
)
from ._plot import MAX_GROUPS
from ._utils import logger

#: The samples `depth-plot` uses: `sample_id`, a dense `sample_idx` in
#: `sample_id` order, and `group_name`.
ROSTER_TABLE = "depth_roster"

#: The genomes `depth-plot` draws: `genome_id`, `length`, `is_circular`.
GENOMES_TABLE = "depth_genomes"


def _ids(con, sql):
    return {row[0] for row in con.sql(sql).fetchall()}


def _report(ids, what):
    if ids:
        ordered = sorted(ids)
        logger.warning(
            f"{len(ordered)} {what}, so are left out: "
            f"{_examples([(i,) for i in ordered], limit=len(ordered))}"
        )


def intersect_layers(con, depth_view, breadth_view, *, orfs):
    """Settle which samples and genomes `depth-plot` uses.

    A sample is used if it is in the metadata and has an aligned read in both
    layers; a genome, if it is in the features and has an aligned read in both
    layers. Everything the metadata or features name but this leaves out is
    reported by name. What they do not name -- a sample with no metadata, a
    genome not asked for -- was left out by the user, and is not.

    Requires `SAMPLE_GROUPS_TABLE`, `DEPTH_FEATURES_TABLE` and the two layer
    views, and `ORFS_TABLE` if `orfs`. Creates `ROSTER_TABLE` and
    `GENOMES_TABLE`.

    Raises
    ------
    ValueError
        If no sample or no genome is left, if more than `MAX_GROUPS` groups
        are, if an aligned read starts beyond its genome's length, or if
        `orfs` and no ORF is on a genome that is left.
    """
    def aligned(view, column):
        return _ids(
            con, f"SELECT DISTINCT {column} FROM {view} WHERE {ALIGNED_ROWS}"
        )

    listed = _ids(con, f"SELECT {COLUMN_SAMPLE_ID} FROM {SAMPLE_GROUPS_TABLE}")
    depth = aligned(depth_view, COLUMN_SAMPLE_ID)
    breadth = aligned(breadth_view, COLUMN_SAMPLE_ID)
    samples = listed & depth & breadth
    if not samples:
        raise ValueError(
            "No sample is in the sample metadata and has alignments in both the "
            f"depth and breadth layers (metadata {len(listed)}, depth "
            f"{len(depth)}, breadth {len(breadth)})."
        )
    _report((listed & depth) - breadth, "sample(s) are in the depth layer only")
    _report((listed & breadth) - depth, "sample(s) are in the breadth layer only")
    _report(listed - depth - breadth, "sample(s) have no alignments in either layer")

    features = _ids(con, f"SELECT {COLUMN_GENOME_ID} FROM {DEPTH_FEATURES_TABLE}")
    depth = aligned(depth_view, "reference")
    breadth = aligned(breadth_view, "reference")
    genomes = features & depth & breadth
    if not genomes:
        raise ValueError(
            "No genome is in the features and has alignments in both the depth "
            f"and breadth layers (features {len(features)}, depth {len(depth)}, "
            f"breadth {len(breadth)})."
        )
    _report((features & depth) - breadth, "genome(s) are in the depth layer only")
    _report((features & breadth) - depth, "genome(s) are in the breadth layer only")
    _report(features - depth - breadth, "genome(s) have no alignments in either layer")

    con.execute(
        f"""CREATE OR REPLACE TABLE {ROSTER_TABLE} AS
            SELECT {COLUMN_SAMPLE_ID},
                   (row_number() OVER (ORDER BY {COLUMN_SAMPLE_ID}) - 1)::INTEGER
                       AS sample_idx,
                   group_name
            FROM {SAMPLE_GROUPS_TABLE}
            WHERE list_contains(?, {COLUMN_SAMPLE_ID})""",
        [sorted(samples)],
    )
    con.execute(
        f"""CREATE OR REPLACE TABLE {GENOMES_TABLE} AS
            SELECT DISTINCT {COLUMN_GENOME_ID}, {COLUMN_LENGTH}, is_circular
            FROM {DEPTH_FEATURES_TABLE}
            WHERE list_contains(?, {COLUMN_GENOME_ID})""",
        [sorted(genomes)],
    )

    groups = sorted(_ids(con, f"SELECT group_name FROM {ROSTER_TABLE}"))
    if len(groups) > MAX_GROUPS:
        raise ValueError(
            f"depth-plot draws at most {MAX_GROUPS} groups, and the metadata "
            f"column has {len(groups)}: {', '.join(groups)}"
        )

    for view in (depth_view, breadth_view):
        rows = con.sql(f"""SELECT a.reference || ' (a read at '
                                      || max(a.position) || ', length '
                                      || g.{COLUMN_LENGTH} || ')'
                           FROM {view} a
                               JOIN {GENOMES_TABLE} g
                                   ON a.reference = g.{COLUMN_GENOME_ID}
                           WHERE {ALIGNED_ROWS}
                               AND a.position > g.{COLUMN_LENGTH}
                           GROUP BY a.reference, g.{COLUMN_LENGTH}
                           ORDER BY 1""").fetchall()
        if rows:
            raise ValueError(
                "An aligned read starts beyond its genome's length, so the "
                f"length in the features is wrong: {_examples(rows)}"
            )

    if orfs:
        on_genomes = con.sql(f"""SELECT count(*) FROM {ORFS_TABLE}
                                 JOIN {GENOMES_TABLE} USING ({COLUMN_GENOME_ID})
                              """).fetchone()[0]
        if on_genomes == 0:
            seqids = con.sql(f"""SELECT DISTINCT {COLUMN_GENOME_ID} FROM {ORFS_TABLE}
                                 ORDER BY 1""").fetchall()
            raise ValueError(
                "No ORF is on a genome being plotted; the ORFs' seqids are "
                f"{_examples(seqids)}. A seqid must equal the genome_id."
            )
