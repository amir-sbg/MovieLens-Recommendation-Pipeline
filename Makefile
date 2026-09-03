.PHONY: test run smoke

test:
	python -m pytest -q

run:
	python -m recsys_lab.cli run

smoke:
	python -m recsys_lab.cli run --users 40 --items 60 --density 0.10 --epochs 4 --top-k 5
