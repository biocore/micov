.PHONY: lint test

test:
	pytest micov

lint:
	ruff check micov setup.py
	check-manifest
