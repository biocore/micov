.PHONY: lint test

test:
	pytest micov
	bash cli_test.sh
lint:
	ruff check micov setup.py
	check-manifest
