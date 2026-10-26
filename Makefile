.PHONY: install run test lint clean

install:
	python -m pip install -r requirements.txt

run:
	python -m insurance_costs.pipeline

test:
	pytest -q

lint:
	python -m compileall src app

clean:
	rm -rf reports/generated/* models/* artifacts/*
