PYTHON ?= python
.PHONY: install inspect test ghz h2 memory-plot render all quick clean

install:
	$(PYTHON) -m pip install -r requirements.txt

inspect:
	$(PYTHON) -m src.check_api

test:
	$(PYTHON) -m pytest -q

ghz:
	$(PYTHON) -m src.ghz_benchmark --max-qubits 30

h2:
	$(PYTHON) -m src.h2_vqe

memory-plot:
	$(PYTHON) -m src.memory_plot

render:
	$(PYTHON) -m src.build_readme

all: install inspect test memory-plot ghz h2 render

quick: memory-plot
	$(PYTHON) -m src.check_api
	$(PYTHON) -m pytest -q -m 'not slow'
	$(PYTHON) -m src.ghz_benchmark --quick
	$(PYTHON) -m src.h2_vqe --quick
	$(PYTHON) -m src.build_readme

clean:
	rm -f results/*.csv results/ghz_*.png results/h2_*.png results/vqe_*.png results/*.json
