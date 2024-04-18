.PHONY: lint test

test:
	pytest micov
	sh usage_tests.sh

lint:
	ruff check micov setup.py
	check-manifest
