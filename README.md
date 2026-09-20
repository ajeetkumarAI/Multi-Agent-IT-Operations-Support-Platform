# Multi-Agent-IT-Operations-Support-Platform
Built a multi-agent AI platform for BFSI support operations that classifies customer requests, gathers missing information, retrieves relevant knowledge, recommends automated resolution steps, and escalates unresolved or sensitive cases to the right human team.

The orchestration framework is **LangGraph**. The main orchestrator coordinates independent LangGraph subagents for each support task.

## Included workflow

The repository includes a modular Python implementation under `src/support_platform` with specialized agents for:

- customer identity validation
- intent classification
- intent confirmation and correction
- missing-information follow-up
- knowledge retrieval
- retrieved-knowledge validation and bounded retry
- automated resolution recommendation
- intelligent human escalation and handoff

## Source layout

```text
src/support_platform/
	models.py
	agent.py            Main LangGraph orchestrator agent
	subagents/
		identity/         LangGraph identity agent + prompt contract
		intent/           LangGraph intent agents + prompt contract
		validation/       LangGraph validation agent + prompt contract
		knowledge/        LangGraph retrieval/validation agents + prompt contract
		resolution/       LangGraph resolution agent + prompt contract
		escalation/       LangGraph escalation agent + prompt contract
```

The root `support_platform.py` file remains as a compatibility import for existing callers. Knowledge retrieval uses SQLite through `KnowledgeRepository`; set `SUPPORT_PLATFORM_DATABASE` to a file path for persistence.

## OpenAI configuration

OpenAI prompt execution is optional. Install dependencies, then configure:

```powershell
$env:SUPPORT_PLATFORM_LLM_ENABLED = "true"
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "gpt-4o-mini"
```

Without those variables, deterministic validation and intent rules remain active. No API call is made during tests unless an LLM is explicitly injected.

## Run tests

```bash
python -m unittest discover -s tests
```

## Run the operations QA app

```powershell
streamlit run app.py
```

The primary app workflow queries a backend-maintained SQLite corpus. It answers operational questions such as SafeWord activation, CMP request creation, issue resolution, installation, POC lookup, and approval routing. Each answer shows procedures, POC, approval rules, record ID, and source metadata. No file upload is required.

The maintained extension corpus is stored in [data/operations_sop_faq.json](data/operations_sop_faq.json). `CorpusRepository.seed()` loads those SOP/FAQ records into SQLite, while `MainOrchestratorAgent.answer_operational_question()` owns retrieval, grounding, and response generation for the UI.

The corpus is represented by `CorpusRepository` and can be replaced by a production metadata API or database without changing the QA interface. OpenAI grounded answers are optional; without an API key the app returns a deterministic answer from the best matching corpus record.

The structured support workflow remains available in the secondary **Support workflow** tab.
