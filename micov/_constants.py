COLUMN_GENOME_ID = 'genome_id'
COLUMN_GENOME_ID_DTYPE = str
COLUMN_SAMPLE_ID = 'sample_id'
COLUMN_SAMPLE_ID_DTYPE = str
COLUMN_START = 'start'
COLUMN_START_DTYPE = int
COLUMN_STOP = 'stop'
COLUMN_STOP_DTYPE = int
COLUMN_READ_ID = 'read'
COLUMN_READ_ID_DTYPE = str
COLUMN_FLAG = 'flag'
COLUMN_FLAG_DTYPE = int
COLUMN_CIGAR = 'cigar'
COLUMN_CIGAR_DTYPE = str
COLUMN_LENGTH = 'length'
COLUMN_LENGTH_DTYPE = int
COLUMN_COVERED = 'covered'
COLUMN_PERCENT_COVERED = 'percent_covered'
COLUMN_COVERED_DTYPE = int
COLUMN_PERCENT_COVERED = 'percent_covered'
COLUMN_PERCENT_COVERED_DTYPE = float

### should really probably just use a dataclass, and type annotations?

class _SCHEMA:
    def __init__(self):
        self.dtypes_dict = dict(self.dtypes_flat)
        self.columns = tuple([c for c, _ in self.dtypes_flat])


class _BED_COV_SCHEMA(_SCHEMA):
    dtypes_flat = [(COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                   (COLUMN_START, COLUMN_START_DTYPE),
                   (COLUMN_STOP, COLUMN_STOP_DTYPE)]
BED_COV_SCHEMA = _BED_COV_SCHEMA()


class _SAM_SUBSET_SCHEMA(_SCHEMA):
    # we only need specific columns, so let's disregard things we are not
    # concerned about.
    # for binary coverage, we don't care about the flag, but we're parsing it
    # now so we can care in the future.
    column_indices = [0, 1, 2, 3, 5]
    dtypes_flat = [(COLUMN_READ_ID, COLUMN_READ_ID_DTYPE),
                   (COLUMN_FLAG, COLUMN_FLAG_DTYPE),
                   (COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                   (COLUMN_START, COLUMN_START_DTYPE),
                   (COLUMN_CIGAR, COLUMN_CIGAR_DTYPE)]
SAM_SUBSET_SCHEMA = _SAM_SUBSET_SCHEMA()


class _SAM_SUBSET_SCHEMA_PARSED(_SCHEMA):
    dtypes_flat = [(COLUMN_READ_ID, COLUMN_READ_ID_DTYPE),
                   (COLUMN_FLAG, COLUMN_FLAG_DTYPE),
                   (COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                   (COLUMN_START, COLUMN_START_DTYPE),
                   (COLUMN_CIGAR, COLUMN_CIGAR_DTYPE),
                   (COLUMN_STOP, COLUMN_STOP_DTYPE)]
SAM_SUBSET_SCHEMA_PARSED = _SAM_SUBSET_SCHEMA_PARSED()


class _GENOME_LENGTH_SCHEMA(_SCHEMA):
    dtypes_flat = [(COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                   (COLUMN_LENGTH, COLUMN_LENGTH_DTYPE)]
GENOME_LENGTH_SCHEMA = _GENOME_LENGTH_SCHEMA()


class _GENOME_COVERAGE_SCHEMA(_SCHEMA):
    dtypes_flat = [(COLUMN_GENOME_ID, COLUMN_GENOME_ID_DTYPE),
                   (COLUMN_COVERED, COLUMN_COVERED_DTYPE),
                   (COLUMN_LENGTH, COLUMN_LENGTH_DTYPE),
                   (COLUMN_PERCENT_COVERED, COLUMN_PERCENT_COVERED_DTYPE)]
GENOME_COVERAGE_SCHEMA = _GENOME_COVERAGE_SCHEMA()
