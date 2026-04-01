# Research Copilot

English | [简体中文](./README.zh-CN.md)

`research_copilot` is a local-first research assistant for working with academic papers from the command line.

It is built for a practical workflow:

- ingest papers into a local store
- build structured paper memory for later retrieval
- ask grounded questions about a paper
- scan a topic using search metadata plus local paper memories
- keep an evolving topic history
- generate a daily digest of recent papers

The project is intentionally lightweight. It uses Python, SQLite, local files under `data/`, and direct OpenAI API calls. It does not use LangChain, a vector database, or a large orchestration framework.

## Table of Contents

- [What This Project Is](#what-this-project-is)
- [Key Components](#key-components)
- [How It Works](#how-it-works)
- [Setup](#setup)
- [How To Use](#how-to-use)
  - [Ingest a paper](#ingest-a-paper)
  - [List stored papers](#list-stored-papers)
  - [Search stored papers](#search-stored-papers)
  - [Build paper memory](#build-paper-memory)
  - [Ask questions about a paper](#ask-questions-about-a-paper)
  - [Run topic scan](#run-topic-scan)
  - [Generate a daily digest](#generate-a-daily-digest)
- [Topic Report Output](#topic-report-output)
- [Design Philosophy](#design-philosophy)
- [Limitations](#limitations)
- [Cost Considerations](#cost-considerations)
- [Future Improvements](#future-improvements)
- [Tests](#tests)

<a id="what-this-project-is"></a>
## What This Project Is

This system is designed to help with two related research problems:

1. Understanding individual papers well enough to ask useful follow-up questions.
2. Maintaining an evolving view of a research area without losing connection to specific papers.

The project separates these concerns:

- **Paper workflow**: PDF/arXiv ingest -> text chunks -> structured paper memory -> grounded QA.
- **Topic workflow**: literature search -> candidate metadata + local memories -> structured topic report -> versioned topic memory.
- **Digest workflow**: recent-paper search over topics -> metadata triage -> daily digest.

It is best thought of as a local research workbench, not as an autonomous research agent.

<a id="key-components"></a>
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

The report includes subthemes, representative papers, recent trends, open questions, evidence quality notes, missing directions, a research-frontier summary, retrieval statistics, and a full appendix of retrieved candidates.

Topic scan now runs in two distinct grounding modes:

- **metadata-only mode**: only search metadata is available, so the report stays conservative about method details and limitations
- **memory-backed mode**: local paper memories are available for part of the candidate set, so the report can make deeper cross-paper comparisons

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

<a id="how-it-works"></a>
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

<a id="setup"></a>
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

<a id="how-to-use"></a>
## How To Use

The primary interface is the CLI:

```bash
python cli.py --help
```

<a id="ingest-a-paper"></a>
### Ingest a paper

From arXiv:

```bash
python cli.py ingest 1706.03762
python cli.py ingest https://arxiv.org/abs/1706.03762
python cli.py ingest 1706.03762 --with-memory
```

From a local PDF:

```bash
python cli.py ingest /path/to/paper.pdf
```

This stores metadata in SQLite, writes extracted text to `data/papers/`, and stores overlapping chunks for later retrieval.

By default, `ingest` stays a cheap non-LLM operation. It now prints the `paper_id` plus suggested next commands so you can immediately build memory, ask a question later, or find the paper again from the local store.

If you want the one-shot flow, use:

```bash
python cli.py ingest 1706.03762 --with-memory
```

That runs ingest first, then builds structured paper memory as a second step.

<a id="list-stored-papers"></a>
### List stored papers

```bash
python cli.py papers
python cli.py papers --limit 100
```

This prints locally stored papers with:

- `paper_id`
- whether memory has already been built
- external id (for arXiv papers, the arXiv id)
- year
- title

<a id="search-stored-papers"></a>
### Search stored papers

```bash
python cli.py search-papers attention
python cli.py search-papers transformer
python cli.py search-papers 1706.03762
```

This searches local papers by title text, keyword, author text, venue/source text, and external id.

<a id="build-paper-memory"></a>
### Build paper memory

```bash
python cli.py memory 1
```

`1` is the `papers.id` returned by ingest.

This runs the paper-memory prompt over the ingested paper text and stores the resulting JSON in both SQLite and `data/cache/`.

<a id="ask-questions-about-a-paper"></a>
### Ask questions about a paper

```bash
python cli.py ask 1 "What is the main contribution?"
python cli.py ask 1 "Why does the method work?"
```

This uses stored paper memory plus retrieved text chunks from the same paper.

If you do not remember the numeric `paper_id`, use `python cli.py papers` or `python cli.py search-papers ...` first.

<a id="run-topic-scan"></a>
### Run topic scan

```bash
python cli.py topic "diffusion models for video" --max-papers 25
```

This:

- searches arXiv and Semantic Scholar
- expands and normalizes the topic into multiple subqueries before merging results
- finds locally relevant paper memories
- builds a structured topic report
- writes Markdown under `data/topics/`
- writes a JSON cache under `data/cache/`
- stores a versioned report snapshot in SQLite when quality heuristics allow persistence

### Use high-quality mode

```bash
python cli.py topic "test-time compute" --max-papers 30 --high-quality
```

`--high-quality` forces the second topic-scan refinement pass even if the heuristic router would otherwise skip it.

### Force refresh topic scan

```bash
python cli.py topic "4d object reconstruction / generation" --max-papers 20 --force
```

`--force` bypasses the topic JSON cache for that `(topic, max_papers)` pair and reruns retrieval plus LLM synthesis. Without `--force`, the CLI reuses the cached topic report when the cache version, topic string, and `--max-papers` value match.

<a id="topic-report-output"></a>
## Topic Report Output

Each topic scan writes:

- `data/topics/<slug>_report.md`: human-readable Markdown report
- `data/cache/topic_report_<slug>.json`: JSON cache with the structured report, retrieval summary, and retrieved candidate appendix

### Markdown sections

The topic report Markdown includes:

- **Summary**: concise landscape summary
- **Scan mode**: whether the report is metadata-only or memory-backed
- **Retrieval stats**: arXiv count, Semantic Scholar count, deduped count, retained candidate count, final report mention count, and provider failures / rate limits
- **Evidence quality**: explicit confidence and coverage limits
- **Branches / subthemes**: major clusters in the topic
- **Paper lists**: foundational, representative, recent valuable, lower-priority / possibly overhyped
- **Recent trends / Open questions**: grounded synthesis from the retrieved set
- **All retrieved candidate papers**: appendix listing every retained candidate with title, year, source(s), URL, and arXiv id when available

### Cache behavior

Topic scan caching is per topic slug and `--max-papers` value. The JSON cache stores:

- the rendered structured topic report
- retrieval query variants and provider stats
- the retained candidate-paper list used for the report appendix

If the report fails the SQLite persistence quality gate, Markdown and JSON cache files are still written; only the topic-memory snapshot is skipped.

### Provider degraded mode and rate limits

- If one provider succeeds and the other fails or is rate-limited, topic scan continues in degraded mode and records that status in the Markdown retrieval section and JSON cache.
- If both providers effectively fail and the failure is due to rate limiting, topic scan raises a rate-limit error.
- Semantic Scholar throttling is still preserved internally; adding `SEMANTIC_SCHOLAR_API_KEY` improves quota and reduces degraded-mode scans.

<a id="generate-a-daily-digest"></a>
### Generate a daily digest

`digest` is the recent-paper triage workflow. It searches for papers published within a recent time window, groups them by one or more topics, asks the LLM to produce a concise recommendation table, and writes the result to `data/digests/`.

CLI form:

```bash
python cli.py digest [topics ...] [--days N] [--max-per-topic N]
```

Arguments:

- `topics`: zero or more topic strings
- `--days`: publication window in days; default `3`
- `--max-per-topic`: max retrieved papers per topic before merge and LLM triage; default `15`

Behavior:

- If one or more `topics` are provided, digest runs on exactly those topics.
- If no `topics` are provided, digest falls back to active subscriptions from SQLite.
- The command prints the digest Markdown path to stderr and prints `items: <count>` to stdout.

Important distinction:

- `python cli.py subscriptions` shows subscribed digest topics, not papers.
- `python cli.py digest ...` performs the recent-paper search and generates the paper list in Markdown.

`subscriptions` output columns are:

- column 1: topic slug, such as `animal_dataset`
- column 2: original topic text, such as `animal dataset`
- column 3: status, usually `active` or `inactive`

So this:

```bash
python cli.py subscriptions
```

means "which topics will be used when I run `digest` without explicit topics?", not "which papers were found already?"

Run digest with explicit topics:

```bash
python cli.py digest --days 5 "reasoning models" "retrieval augmented generation"
```

Equivalent examples:

```bash
python cli.py digest "animal motion"
python cli.py digest "animal motion" "4d generation" --days 7 --max-per-topic 20
```

Or subscribe topics once and reuse them when no topics are passed:

```bash
python cli.py subscribe "reasoning models"
python cli.py subscribe "multimodal agents"
python cli.py subscriptions
python cli.py digest --days 3
```

Example workflow:

1. Subscribe one or more standing topics:

```bash
python cli.py subscribe "animal dataset"
python cli.py subscribe "animal motion"
```

2. Check which topics are currently active:

```bash
python cli.py subscriptions
```

Example output:

```text
animal_dataset  animal dataset  active
animal_motion   animal motion   active
```

This does not mean papers have already been retrieved. It only means those topics are saved as digest inputs.

3. Run digest without passing explicit topics:

```bash
python cli.py digest --days 3
```

This uses the active subscriptions above, searches for papers from the last 3 days, and writes a digest Markdown file under `data/digests/`.

4. Open the generated digest Markdown to see the actual paper recommendations.

If you want to bypass subscriptions and search directly, pass topics explicitly:

```bash
python cli.py digest --days 3 "animal dataset"
python cli.py digest --days 7 "animal dataset" "animal motion"
```

What digest does internally:

- searches recent papers for each topic
- merges and deduplicates overlapping results across topics
- sends candidate metadata to the digest prompt
- writes a Markdown digest under `data/digests/`

What digest is not:

- not a topic survey like `python cli.py topic ...`
- not a full-paper review
- not based on local paper memory by default

Use it as a lightweight filter for new papers worth reading next.

<a id="design-philosophy"></a>
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

<a id="limitations"></a>
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

<a id="cost-considerations"></a>
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

<a id="future-improvements"></a>
## Future Improvements

- stronger retrieval for paper QA and topic-memory matching
- better validation between topic claims and supporting source papers
- richer evidence tracking from paper memory into topic reports
- more explicit confidence fields in topic reports
- better handling of poor PDF extraction and incomplete metadata

<a id="tests"></a>
## Tests

```bash
python -m unittest discover -s tests -v
```

The current tests focus on parsing, chunking, database behavior, retrieval helpers, CLI wiring, and digest Markdown rendering. They do not fully validate research correctness.
