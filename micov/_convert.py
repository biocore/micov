from functools import lru_cache


@lru_cache(maxsize=128)
def cigar_to_lens(cigar):
    """Extract lengths from a CIGAR string.

    Adapted from woltka (woltka.align.cigar_to_lens) which is a BSD-3 codebase:

    https://github.com/qiyunzhu/woltka/blob/09bfef673be34825dcb36f6d5c43442dd0f1e88c/woltka/align.py#L613

    Parameters
    ----------
    cigar : str
        CIGAR string.

    Returns
    -------
    int
        Alignment length.
    int
        Offset in subject sequence.

    Notes
    -----
    This function significantly benefits from LRU cache because high-frequency
    CIGAR strings (e.g., "150M") are common and redundant calculations can be
    saved.

    """
    align, offset = 0, 0
    n = ''  # current step size
    for c in cigar:
        if c in 'MDIHNPSX=':
            if c in 'M=X':
                align += int(n)
            elif c in 'DN':
                offset += int(n)
            n = ''
        else:
            n += c

    # see https://replicongenetics.com/cigar-strings-explained/
    # we express coverage as the range of positions in the reference
    # covered by the read, inclusive of deletions and gaps in the query
    return align + offset
