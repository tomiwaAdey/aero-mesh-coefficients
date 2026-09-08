PYTHON ?= python3
PYTHONPATH := src:$(PYTHONPATH)

.PHONY: test fetch-avl fetch-gmsh fetch-onera fetch-crm prepare-external verify-sources run-pipeline run-model-selection run-solver-validation run-structured-comparator run-statistics run-external-validation run-wake-appendix run-stl-robustness summarize paper-figures paper reproduce all

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

fetch-avl:
	scripts/fetch_avl_reference.sh

fetch-gmsh:
	scripts/fetch_gmsh_reference.sh

fetch-onera:
	scripts/fetch_onera_m6_reference.sh

fetch-crm:
	scripts/fetch_nasa_crm_reference.sh

prepare-external: fetch-avl fetch-gmsh fetch-onera fetch-crm
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/mesh_nasa_crm_reference.py

verify-sources:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/verify_source_manifest.py --require-tools

run-pipeline:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/run_pipeline.py

run-model-selection:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/run_model_selection.py

run-solver-validation:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/run_solver_validation.py

run-structured-comparator:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/run_structured_comparator.py

run-statistics:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/run_statistical_analysis.py

run-external-validation:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/run_external_validation.py

run-wake-appendix:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/run_wake_appendix.py

run-stl-robustness:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/run_stl_robustness.py

summarize:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_summary.py

paper-figures:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_paper_diagrams.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_research_overview.py

paper:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/build_paper.py

reproduce: prepare-external verify-sources test run-pipeline run-model-selection run-solver-validation run-structured-comparator run-statistics run-external-validation run-wake-appendix run-stl-robustness summarize paper-figures paper

all: test paper
