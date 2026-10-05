.PHONY: lint lint-fix test

test:
	pytest micov
	bash cli_test.sh
lint:
	ruff check micov
	check-manifest

# `lint` only reports. Rewriting files is opt-in, so a newly enabled rule can
# never silently change production code during a check.
lint-fix:
	ruff check --fix micov
