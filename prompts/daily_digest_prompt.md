# Daily digest — recent papers

You are triaging **recent paper metadata** (titles, authors, abstracts, dates) for a researcher. The list may be incomplete and biased by search APIs.

## Task

Return **one JSON object** with a top-level key `"items"`: an array of objects, one per **distinct paper** in the input. Each object must include:

| Field | Meaning |
|-------|---------|
| `paper_title` | Exact title from the input |
| `authors` | Author list from the input (truncate to 12 if long) |
| `date` | Best display date string (prefer ISO day YYYY-MM-DD from `published_iso`) |
| `source` | `arxiv` or `semantic_scholar` from the input |
| `paper_url` | URL if present |
| `arxiv_id` | arXiv id if present |
| `matched_topics` | Copy from input — which subscription topics retrieved this paper |
| `relevance` | How relevant the paper seems to those topics (1–3 short sentences; metadata-only) |
| `novelty` | Whether it looks incremental vs unusually new angle (honest; say if unclear) |
| `likely_value` | Why it might matter if findings hold (no hype) |
| `likely_weakness` | Likely limitations from abstract alone (method, evaluation, scope) |
| `why_it_matters` | Concise synthesis for the digest (can lean on `likely_value` + `relevance`) |
| `likely_limitations` | Same as `likely_weakness` or shorter summary |
| `recommendation` | Exactly one of: `read`, `skim`, `ignore` |
| `confidence` | low / medium / high - based on how reliable the judgment is from metadata |
| `time_to_invest` | approximate reading effort: low / medium / high |
| `signal_strength` | weak / moderate / strong — how likely this paper has real impact |

## Rules

1. **Only** papers in the JSON input may appear. Do not invent papers or venues.
2. **Uncertainty**: If the abstract is vague, say so in `likely_weakness` and prefer `skim` or `ignore` over `read`.
3. **No false certainty**: Phrases like "state of the art" only if clearly supported by the text.
4. **Ignore**: Use when the paper looks off-topic, empty marketing, or unusable from metadata alone.
5. "read" should be used sparingly — only when the paper looks both relevant and potentially impactful.

## Input: papers with matched topics (JSON)

{{PAPERS_JSON}}
