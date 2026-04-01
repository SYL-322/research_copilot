# Research Copilot

`research_copilot` is a local-first research assistant for working with academic papers from the command line.

It is built for a practical workflow:

- ingest papers into a local store
- build structured paper memory for later retrieval
- ask grounded questions about a paper
- scan a topic using search metadata plus local paper memories
- keep an evolving topic history
- generate a daily digest of recent papers

The project is intentionally lightweight. It uses Python, SQLite, local files under `data/`, and direct OpenAI API calls. It does not use LangChain, a vector database, or a large orchestration framework.

## What This Project Is

This system is designed to help with two related research problems:

1. Understanding individual papers well enough to ask useful follow-up questions.
2. Maintaining an evolving view of a research area without losing connection to specific papers.

The project separates these concerns:

- **Paper workflow**: PDF/arXiv ingest -> text chunks -> structured paper memory -> grounded QA.
- **Topic workflow**: literature search -> candidate metadata + local memories -> structured topic report -> versioned topic memory.
- **Digest workflow**: recent-paper search over topics -> metadata triage -> daily digest.

It is best thought of as a local research workbench, not as an autonomous research agent.

## Key Components

### Paper memory

Paper memory is a structured JSON summary of one paper, generated from ingested paper text with an LLM. The schema includes:

- problem
- core idea
- method overview
- key components
- design rationale
- assumptions
- contributions
- limitations
- glossary
- failure modes
- open questions

The intent is to produce a reusable, compressed representation of a paper that is more structured than a plain summary and more useful than storing raw chunks alone.

If a long paper is truncated before being sent to the LLM, the stored memory includes `truncated: true` so downstream steps can treat it as potentially incomplete.

### Grounded QA

Paper QA answers questions about one paper using two sources:

- the stored paper memory
- retrieved chunks from the paper text

The prompt instructs the model to use retrieved excerpts as primary evidence and cite `chunk_index` values when relying on them. This is meant to keep answers tied to local evidence rather than free-form recollection.

If no supporting chunks are retrieved for a question, QA now explicitly marks that condition in the prompt and answers more conservatively from paper memory alone.

### Topic scanning

Topic scan builds a structured report for a research topic from:

- arXiv search results
- Semantic Scholar search results
- locally stored paper memories that lexically match the topic
- a short summary of prior topic reports

The report includes subthemes, representative papers, recent trends, open questions, evidence quality notes, missing directions, and a research-frontier summary.

### Topic memory

Topic memory is the versioned history of prior topic reports. Each successful topic scan can be stored as a snapshot in SQLite. Later scans pull a short summary from recent snapshots and ask the model to treat those prior claims as hypotheses, not facts.

This is an evolving memory layer, not a ground-truth knowledge base.

### Two-pass topic scan

Topic scan supports a two-pass flow:

- pass 1 uses the light model
- pass 2 runs only when quality heuristics or complexity heuristics trigger refinement, or when `--high-quality` is used

The second pass receives the initial JSON report and is asked to improve comparison, gaps, and frontier analysis.

### Model routing

The project distinguishes between:

- **main model**: `OPENAI_MODEL`
- **light model**: `OPENAI_MODEL_LIGHT`, falling back to the main model when unset

Paper memory and paper QA use the main model directly. Topic scan routes between light and main models. This is the main built-in quality/cost control mechanism.

## How It Works

### System overview

The main data flow is:

`paper -> chunks -> paper memory -> topic report -> topic memory -> daily digest`

In more detail:

1. A paper is ingested from a local PDF or arXiv id/URL.
2. Text is extracted and split into overlapping chunks.
3. Paper metadata and chunks are stored in SQLite; extracted text is also written to disk.
4. A paper-memory prompt turns the paper text into a structured JSON memory.
5. Paper QA uses that memory plus retrieved chunks to answer questions.
6. Topic scan searches external APIs for candidate papers, then augments those candidates with locally relevant paper memories.
7. Topic scan writes a Markdown report and stores a versioned JSON snapshot for future topic memory.
8. Daily digest searches recent papers for one or more topics and asks the LLM to triage them into a concise digest.

### Storage layout

By default, the system writes under `./data/`:

- `data/research_copilot.db`: SQLite database
- `data/papers/`: local PDFs and extracted text
- `data/topics/`: rendered topic reports
- `data/digests/`: rendered digests
- `data/cache/`: JSON cache artifacts for ingest, paper memory, and topic reports

### Database entities

The main SQLite tables are:

- `papers`
- `paper_chunks`
- `paper_memories`
- `topic_reports`
- `topic_report_versions`
- `subscriptions`
- `daily_digests`

The schema is deliberately simple so the local data remains inspectable and portable.

## Setup

```bash
cd research_copilot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set at least:

```bash
OPENAI_API_KEY=...
```

Optional configuration:

- `OPENAI_MODEL`
- `OPENAI_MODEL_LIGHT`
- `SEMANTIC_SCHOLAR_API_KEY`
- `RESEARCH_COPILOT_DATA_DIR`
- `DATABASE_FILENAME`
- `HTTP_TIMEOUT`
- `LOG_LEVEL`

## How To Use

The primary interface is the CLI:

```bash
python cli.py --help
```

### Ingest a paper

From arXiv:

```bash
python cli.py ingest 1706.03762
python cli.py ingest https://arxiv.org/abs/1706.03762
```

From a local PDF:

```bash
python cli.py ingest /path/to/paper.pdf
```

This stores metadata in SQLite, writes extracted text to `data/papers/`, and stores overlapping chunks for later retrieval.

### Build paper memory

```bash
python cli.py memory 1
```

`1` is the `papers.id` returned by ingest.

This runs the paper-memory prompt over the ingested paper text and stores the resulting JSON in both SQLite and `data/cache/`.

### Ask questions about a paper

```bash
python cli.py ask 1 "What is the main contribution?"
python cli.py ask 1 "Why does the method work?"
```

This uses stored paper memory plus retrieved text chunks from the same paper.

### Run topic scan

```bash
python cli.py topic "diffusion models for video" --max-papers 25
```

This:

- searches arXiv and Semantic Scholar
- finds locally relevant paper memories
- builds a structured topic report
- writes Markdown under `data/topics/`
- stores a versioned report snapshot in SQLite when quality heuristics allow persistence

### Use high-quality mode

```bash
python cli.py topic "test-time compute" --max-papers 30 --high-quality
```

`--high-quality` forces the second topic-scan refinement pass even if the heuristic router would otherwise skip it.

### Generate a daily digest

Pass explicit topics:

```bash
python cli.py digest --days 5 "reasoning models" "retrieval augmented generation"
```

Or subscribe topics once and reuse them:

```bash
python cli.py subscribe "reasoning models"
python cli.py subscribe "multimodal agents"
python cli.py subscriptions
python cli.py digest --days 3
```

## Design Philosophy

### Local-first

The system keeps the durable state locally:

- SQLite for structured records
- local files for extracted text, topic reports, digests, and caches

This makes the workflow inspectable, scriptable, and usable over SSH.

### Minimal dependencies

The project uses a small set of direct components:

- Python
- SQLite
- `httpx`
- `pydantic`
- OpenAI SDK
- PDF extraction libraries

The implementation avoids a heavy orchestration layer on purpose.

### Structured outputs

The major LLM steps use Pydantic response schemas:

- paper memory
- topic report
- daily digest

This constrains output shape, simplifies validation, and makes it possible to persist JSON cleanly.

### Avoid hallucination

The system tries to reduce hallucination by:

- grounding paper QA in retrieved chunks
- constraining topic and digest outputs to supplied candidate papers
- explicitly asking the model to admit uncertainty
- validating outputs against expected schemas

These measures reduce risk, but they do not eliminate incorrect reasoning.

### Cost-aware

The system is built around a simple cost strategy:

- use a light model for first-pass topic scan
- escalate to the main model only when needed
- keep retrieval local and lexical
- avoid extra infrastructure such as vector services

## Limitations

These limitations matter for real research use:

### Topic scan is partially metadata-based

Topic scan depends heavily on titles, abstracts, venues, dates, and search coverage from arXiv and Semantic Scholar. Only a subset of topic reasoning may be backed by full local paper memories.

This means:

- paper categorization can be shallow
- trends can reflect search bias
- missing directions may reflect retrieval gaps rather than true gaps in the field

### LLM outputs are not guaranteed correct

Paper memory, QA, topic reports, and digests all rely on LLM judgments. Structured output validation checks format, not truth.

### Topic memory can accumulate bias

Prior topic reports are fed back into later topic scans as short claim summaries. The prompt tells the model to treat them as hypotheses, but this is still a self-referential memory loop and can reinforce earlier mistakes or framing bias.

### Second-pass refinement is heuristic

The second topic pass is triggered by simple heuristics such as report length, number of grounded mentions, prior-memory presence, and explicit `--high-quality`. This helps with cost control, but it is not a robust measure of epistemic quality.

### Retrieval is lexical, not semantic

Paper QA and memory lookup use token-overlap heuristics rather than embeddings or learned ranking. Good evidence can be missed when wording differs.

### Digest is metadata triage, not paper review

Daily digest judgments are based on recent-paper metadata, especially titles and abstracts. They should be treated as triage suggestions, not as reliable assessments of technical quality or empirical validity.

By default, digest generation uses `OPENAI_MODEL_LIGHT` when configured, falling back to `OPENAI_MODEL` otherwise. Digest rows may also include lightweight triage metadata such as `confidence`, `time_to_invest`, and `signal_strength`.

### Ingest is PDF-text dependent

If PDF extraction is poor, the downstream memory and QA quality will degrade accordingly.

## Cost Considerations

The main knobs are:

- `OPENAI_MODEL`: main model for paper memory, paper QA, and topic refinement
- `OPENAI_MODEL_LIGHT`: first-pass topic-scan model and default digest model
- `--high-quality`: forces the second topic pass
- `--max-papers`: controls topic-scan search breadth
- `--days` and `--max-per-topic`: control digest size

Practical guidance:

- keep `OPENAI_MODEL_LIGHT` cheaper than `OPENAI_MODEL`
- use default topic scan first, then rerun with `--high-quality` only when needed
- keep `--max-papers` moderate unless the topic is broad
- use digest as a filter, not as a substitute for reading papers

## Future Improvements

- stronger retrieval for paper QA and topic-memory matching
- better validation between topic claims and supporting source papers
- richer evidence tracking from paper memory into topic reports
- more explicit confidence fields in topic reports
- better handling of poor PDF extraction and incomplete metadata

## Tests

```bash
python -m unittest discover -s tests -v
```

The current tests focus on parsing, chunking, database behavior, retrieval helpers, CLI wiring, and digest Markdown rendering. They do not fully validate research correctness.
