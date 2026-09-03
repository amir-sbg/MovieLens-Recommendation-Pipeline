.PHONY: test run smoke

test:
	python -m pytest -q

run:
	PYTHONPATH=src python -m recsys_lab.cli run

smoke:
	PYTHONPATH=src python -m recsys_lab.cli run --users 40 --items 60 --density 0.10 --mf-epochs 4 --top-k 5
