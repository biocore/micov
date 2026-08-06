micov ChangeLog
=====================

micov 0.0.1-dev
---------------

Backward incompatible changes:

* The `micov per-sample-group` subcommand is now `micov per-sample`. Click 8.2
  merged https://github.com/pallets/click/pull/2604, which implicitly strips
  `_group` suffixes when deriving a command name from its function name, so the
  `per_sample_group` callback registers as `per-sample`. micov previously pinned
  `click<8.2` to hold the old name; that pin has been removed and the shorter
  name is now canonical. `micov per-sample-group` no longer resolves.
