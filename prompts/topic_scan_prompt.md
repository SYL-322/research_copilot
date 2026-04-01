# Topic literature scan

You are helping a researcher understand the landscape of a **research topic** using several sources:

1. **Candidate papers** — metadata only (titles, abstracts, etc.): broad coverage from arXiv and Semantic Scholar.
2. **Previously analyzed paper memories** — structured summaries from our local library: deeper and more reliable for methods, assumptions, and limitations when available.
3. **Prior claims summary** — up to three short bullet points distilled from earlier topic snapshots for this topic (if any).

The candidate list is incomplete; memories may cover only a subset of papers but add depth.

## Prior claims (hypotheses only)

You may be given **PRIOR_CLAIMS_SUMMARY_JSON** — a JSON array of short strings (max 3).

When prior claims exist:

- Treat them as **hypotheses**, not facts. **Only keep or restate** a claim if it is **supported by current candidate metadata and/or paper memories**.
- If a prior claim conflicts with stronger evidence now, **prefer the evidence** and say briefly what changed in **`evolution_notes`**.
- Do **not** treat prior claims as a second source of truth alongside candidates—use them to **spot continuity** and **revision**, not to pad the report.

## How to use each source

- **Paper memories**: Prefer these for deeper **method comparison**, **design rationale**, **failure modes**, and **what actually differs** between approaches. When a memory and a candidate refer to the same paper (match by title, external id, or obvious identity), **prioritize the memory** over abstract-only metadata.
- If a paper memory includes `truncated: true`, treat it as potentially incomplete coverage of that paper and avoid overconfident conclusions from missing details.
- **Candidate papers**: Use for **coverage**, **recency**, **trends**, and papers you have not seen in memory form.
- **Prior claims**: Use sparingly to **avoid repeating** generic points unless you add nuance grounded in current evidence.

## Cross-paper reasoning

Where possible, **compare approaches across multiple papers** using the memories (and metadata where memories are absent). Highlight:

- Key differences in **method design**
- **Recurring assumptions** across work
- **Common failure patterns** or limitations

Populate **`cross_paper_insights`** (short bullet strings) and **`method_comparison_summary`** (one cohesive paragraph) with this synthesis when memories or metadata allow.

## Your task

Produce **one JSON object** matching the required schema. Be explicit when evidence is weak: from metadata alone, say what you can infer from titles/abstracts only, and avoid pretending you have read full papers.

Your JSON **must** include:

- **`missing_directions`**: array of strings — important approaches or sub-areas likely **missing from** the candidate list.
- **`research_frontier`**: string — where the field seems to be going, based on candidates, trends, and memories (if any).
- **`evolution_notes`**: string — how this report **improves, changes, or supersedes** prior topic understanding (if you had prior claims or memories; otherwise briefly state that this is a first pass).

## Refinement pass (when `INITIAL_TOPIC_REPORT_JSON` is not null)

When `INITIAL_TOPIC_REPORT_JSON` contains a prior draft (same schema as your output), this is a **refinement pass**:

- **Do NOT** repeat the initial report or restate its narrative.
- Focus on **deeper comparison**, **better insights**, and **clearer gaps**.
- **Improve** `research_frontier` and `missing_directions` in particular.
- Output **one** complete replacement JSON object.

## Rules

1. **No hallucinated papers**: Every paper you list in **foundational_papers**, **representative_papers**, **recent_valuable_papers**, and **lower_priority_or_overhyped** must correspond to a **candidate** entry (match by title and/or arXiv id / URL). You may *omit* irrelevant candidates. Do not invent authors, venues, or years not present in the candidate JSON.
2. **Overlap with memories**: When the same paper appears in both candidate metadata and paper memories, **prefer paper memories** for substantive claims (methods, assumptions, limitations). Use metadata for bibliographic fields if needed.
3. **Uncertainty**: Use `evidence_quality_note` to state limitations (e.g. search bias, recency skew, few memories for this topic).
4. **Categories** (candidates only for listed papers):
   - **foundational_papers**: Defining or early work *among the candidates* (use `note` to justify).
   - **representative_papers**: Typical or benchmark-setting work *among the candidates*.
   - **recent_valuable_papers**: Recent papers that look substantive from metadata (prefer last ~3–5 years if visible).
   - **lower_priority_or_overhyped**: Candidates that look incremental, vague, or possibly overclaimed from metadata alone—be cautious and brief.
5. **branches_subthemes**: Cluster the topic into named branches; describe each briefly.
6. **recent_trends** and **open_questions**: Synthesis—label speculation clearly if not directly stated in abstracts or memories.
7. **Gaps vs. candidates**: If important directions are missing from the candidate list, describe them **abstractly** in `missing_directions` without inventing paper rows not in the candidate JSON.

## Topic

{{TOPIC}}

## Candidate papers (metadata JSON array)

{{CANDIDATE_PAPERS_JSON}}

## Related paper memories (local structured summaries; may be empty `[]`)

{{RELATED_PAPER_MEMORIES_JSON}}

## Prior claims summary (JSON array of short strings; may be empty `[]`)

{{PRIOR_CLAIMS_SUMMARY_JSON}}

## Initial topic report JSON (`null` or prior draft for refinement)

`INITIAL_TOPIC_REPORT_JSON` is either JSON **`null`** or a full prior report object (same schema as your output).

- If **`null`**: produce a fresh report from the other inputs.
- If a **JSON object**: **refinement pass** — follow the refinement instructions above.

{{INITIAL_TOPIC_REPORT_JSON}}
