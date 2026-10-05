# micov internals

Reference material for agents and contributors changing micov. It describes
the code **as it is on this branch**: DuckDB plus the miint extension, with
no polars, scipy, numba or pyarrow. History appears only where it explains a
constraint that still binds, such as why a golden was produced by scipy.

Each document opens with **Read this when**, so you can pick one without
reading them all. Facts point at code as `module.symbol`, so you can check
them, and the code wins over these docs if they disagree. Fix the doc in the
same change.

| Document | Read this when |
|---|---|
| [architecture.md](architecture.md) | You need the data flow, the module boundaries, or where a piece of logic lives |
| [data-formats.md](data-formats.md) | You touch any input parser or any output writer, or anything else covered by the compatibility contract |
| [commands.md](commands.md) | You change a CLI command, or need to know what one does end to end |
| [view.md](view.md) | You touch `View`, feature filtering, regions, or presence/absence |
| [curves-and-ks.md](curves-and-ks.md) | You touch ranking, cumulative curves, Monte Carlo, position plots, or KS output |
| [miint.md](miint.md) | You add or change a call into the miint extension, or debug loading it |
| [testing.md](testing.md) | You add a test, see a golden fail, or need to know what a tier covers |
| [traps.md](traps.md) | **Before any non-trivial change.** Each entry is a mistake already made once |

## Ground rules these documents assume

- `CLAUDE.md` holds the project rules: TDD, never `rm`, and never change an
  expected test value without permission. These documents do not repeat them.
- **Outputs are contracts.** The CLI command set, the Parquet pair, `.ks.csv`
  and the position `.tsv.gz` are read by released micov and by published
  analyses. A change that moves a published number is a regression.
- When a fact here is about behaviour, a test pins it. Each document names
  the test, and if you change the behaviour, that test should fail first.
