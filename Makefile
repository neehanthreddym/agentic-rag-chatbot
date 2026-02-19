.PHONY: sanity chat test test-unit

sanity:
	@echo "🔍 Running sanity check..."
	@mkdir -p artifacts
	.venv/bin/python -m scripts.run_sanity
	@echo "✅ Sanity check complete — see artifacts/sanity_output.json"

chat:
	@echo "🚀 Launching chat UI..."
	.venv/bin/streamlit run app.py

test:
	.venv/bin/python -m pytest tests/ -v

test-unit:
	.venv/bin/python -m pytest tests/ -v -m "not integration"