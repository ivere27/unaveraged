.DEFAULT_GOAL := build
.DELETE_ON_ERROR:
.NOTPARALLEL:

PYTHON ?= python3
NPM ?= npm
MODEL ?= all
BROWSER ?= all
PRECISION ?= default
QAT_BACKEND ?= cpu
JOBS ?= auto
PORT ?= 8000
ACTION ?= $(if $(strip $(TAG)),create,models)
TAG ?=
REPO ?=
DRAFT ?= 0
CLOBBER ?= 0
DRY_RUN ?= 0
export TAG REPO DRAFT CLOBBER DRY_RUN
export BROWSER PRECISION QAT_BACKEND JOBS

.PHONY: build test demo package release publish clean help deps check-inputs

# Everyday workflow. Build trains/calibrates; test and demo reuse existing ZIPs.
build: check-inputs deps
	.venv/bin/python shared/build.py --model "$(MODEL)" --precision "$(PRECISION)" --browser "$(BROWSER)"
package: deps
	.venv/bin/python shared/build.py --model "$(MODEL)" --precision "$(PRECISION)" --browser "$(BROWSER)" --package-only
demo: .venv/.docs-deps
	.venv/bin/python shared/pages.py
	@printf 'Open http://127.0.0.1:$(PORT)/example/ (Ctrl-C to stop)\n'
	"$(PYTHON)" -m http.server "$(PORT)" --bind 127.0.0.1 --directory dist/pages
release: .venv/.docs-deps
	.venv/bin/python shared/release.py "$(ACTION)"
publish: release
clean:
	rm -rf -- dist
help:
	@printf '%s\n' \
	 'make                         Build both models for Chrome and Firefox (trains/calibrates)' \
	 'make test                    Verify existing packages; no model rebuild' \
	 'make demo [PORT=8000]        Stage existing packages and serve the local example' \
	 'make package                 Refresh packages using cached models' \
	 'make release                 Upload local model packages for Pages; no TAG needed' \
	 'make clean                   Remove dist outputs; keep model caches' \
	 'make help                    Show this list' \
	 '' \
	 'Build/package/test: MODEL=all|digit|vqa  BROWSER=all|chrome|firefox' \
	 'Build: QAT_BACKEND=cuda for Digit GPU training (default CPU can be slow)' \
	 'Build: JOBS=auto reserves two logical CPUs; JOBS=N sets native CPU threads' \
	 'Docker: docker-image, docker-build, docker-test for container builds/tests' \
	 'Release: TAG=vX.Y.Z for a versioned release; ACTION=check to validate only' \
	 'Release options: REPO=owner/repo, DRAFT=1, DRY_RUN=1, ACTION=upload CLOBBER=1' \
	 'Advanced model/export/benchmark tools: see README; legacy aliases remain compatible'

# Shared preparation and verification.
check-inputs:
	"$(PYTHON)" shared/build.py --model "$(MODEL)" --precision "$(PRECISION)" --check-inputs
deps: .venv/.make-deps node_modules/.make-deps
.venv/bin/python:
	"$(PYTHON)" shared/build.py --venv
.venv/.make-deps: requirements.txt docs/requirements.txt .venv/bin/python
	.venv/bin/python -m pip install --disable-pip-version-check --index-url https://pypi.org/simple -r requirements.txt
	touch "$@"
.venv/.docs-deps: docs/requirements.txt .venv/bin/python
	.venv/bin/python -m pip install --disable-pip-version-check --index-url https://pypi.org/simple -r docs/requirements.txt
	touch "$@"
node_modules/.make-deps: package.json package-lock.json
	"$(NPM)" ci --registry=https://registry.npmjs.org --no-audit --no-fund
	touch "$@"
test: deps
	if [ "$(MODEL)" = all ]; then "$(NPM)" test; else node --test shared/tests/*.test.mjs "$(MODEL)"/*.test.mjs; fi
	.venv/bin/python -m unittest discover -s shared/tests -p 'test_*.py'
	if [ "$(MODEL)" = all ] || [ "$(MODEL)" = digit ]; then .venv/bin/python digit/check.py $(if $(filter default,$(PRECISION)),,--precision "$(PRECISION)"); fi
	if [ "$(MODEL)" = all ] || [ "$(MODEL)" = vqa ]; then .venv/bin/python vqa/check.py $(if $(filter default,$(PRECISION)),,--precision "$(PRECISION)"); fi
ifneq ($(BROWSER),firefox)
	if [ -z "$$RECEIPT_CHROME" ]; then "$(NPM)" exec --offline -- puppeteer browsers install chrome; fi
	node shared/tests/browser.mjs --model "$(MODEL)" --precision "$(PRECISION)" --production
	node shared/tests/browser.mjs --model "$(MODEL)" --precision "$(PRECISION)" --lifecycle
endif
ifneq ($(BROWSER),chrome)
	if [ -z "$$RECEIPT_FIREFOX" ]; then "$(NPM)" exec --offline -- puppeteer browsers install firefox; fi
	node shared/tests/firefox-browser.mjs --model "$(MODEL)" --precision "$(PRECISION)"
	node shared/tests/firefox-browser.mjs --model "$(MODEL)" --precision "$(PRECISION)" --lifecycle
endif

# Compatibility for existing scripts. New usage selects MODEL/BROWSER or uses
# the corresponding Python/Node CLI instead of adding more Make target names.
.PHONY: all digit vqa dev fixtures check check-firefox docs pages source release-check release-upload benchmark benchmark-native benchmark-wasm
all: build
check: test
digit vqa:
	$(MAKE) build MODEL=$@
check-firefox:
	$(MAKE) test BROWSER=firefox
dev: check-inputs deps
	.venv/bin/python shared/build.py --model "$(MODEL)" --precision "$(PRECISION)" --browser "$(BROWSER)" --dev
fixtures: deps
	.venv/bin/python shared/build.py --model "$(MODEL)" --fixtures
docs: .venv/.docs-deps
	.venv/bin/python shared/docs.py
pages: .venv/.docs-deps
	.venv/bin/python shared/pages.py
source:
	"$(PYTHON)" shared/build.py --source
release-check:
	$(MAKE) release ACTION=check
release-upload:
	$(MAKE) release ACTION=upload
BENCHMARK_OUT ?= .cache/benchmarks/retest
benchmark: benchmark-native benchmark-wasm
benchmark-native: deps
	.venv/bin/python benchmarks/native.py --out "$(BENCHMARK_OUT)" $(if $(EVAL_ROOT),--evaluation-root "$(EVAL_ROOT)",)
benchmark-wasm: deps
	node benchmarks/browser.mjs --browser "$(BROWSER)" --out "$(BENCHMARK_OUT)"

DOCKER_IMAGE ?= unaveraged-builder
.PHONY: docker-image docker-build docker-test
docker-image:
	docker build -t $(DOCKER_IMAGE) .

docker-build:
	docker run --rm -v "$$(pwd)":/workspace -v "$$HOME/.cache/puppeteer":/home/builder/.cache/puppeteer -w /workspace $(DOCKER_IMAGE) make MODEL=$(MODEL) BROWSER=$(BROWSER) PRECISION=$(PRECISION)

docker-test:
	docker run --rm -v "$$(pwd)":/workspace -v "$$HOME/.cache/puppeteer":/home/builder/.cache/puppeteer -w /workspace $(DOCKER_IMAGE) npm run test:browser

