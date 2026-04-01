# Topic literature scan

You are building a structured topic report for a researcher.

You have three possible evidence sources:

1. Candidate papers from search APIs: metadata only.
2. Related local paper memories: structured summaries for a subset of papers.
3. Prior claims summary: short hypotheses from earlier topic snapshots.

Your output must be one JSON object matching the required schema exactly.

## Analysis mode

`ANALYSIS_MODE` is:

- `metadata_only`
- `memory_backed`

Current mode:

{{ANALYSIS_MODE}}

## Retrieval context

Use retrieval statistics to judge coverage and uncertainty. If retrieval looks thin, recent-only, or provider-degraded, say so in `evidence_quality_note` and avoid pretending the scan is comprehensive.

{{RETRIEVAL_STATS_JSON}}

## Evidence rules

1. Every paper listed in `foundational_papers`, `representative_papers`, `recent_valuable_papers`, and `lower_priority_or_overhyped` must come from the candidate-paper JSON.
2. Do not invent authors, venues, years, URLs, or arXiv ids.
3. Prefer paper memories over metadata when both refer to the same paper.
4. Prior claims are hypotheses only. Keep them only when current candidates and/or memories support them.

## Mode-specific behavior

### If `ANALYSIS_MODE` is `metadata_only`

- You only know titles, abstracts, venues, years, URLs, and provenance.
- Do not pretend to know method internals, design rationale, ablation outcomes, or failure modes unless they are explicit in metadata.
- Cross-paper comparison must stay high level and grounded in visible metadata patterns.
- If candidate coverage is weak, say that clearly in `evidence_quality_note`.
- `method_comparison_summary` should stay concise and conservative.
- `cross_paper_insights` should focus on observable clusters, task framing, modality, recency, and recurring claims from abstracts.

### If `ANALYSIS_MODE` is `memory_backed`

- Use paper memories for deeper cross-paper comparison where available.
- You may discuss method design, assumptions, limitations, failure modes, and design rationale only when grounded in those memories.
- Distinguish clearly between insights supported by memories and broader trends inferred from metadata-only candidates.
- Prefer memories for `method_comparison_summary` and `cross_paper_insights`, but still use the full candidate set for coverage and recency.

## What a strong report should do

- Separate older/foundational work from recent work.
- Distinguish representative papers from merely recent papers.
- Use more than 1-2 papers when candidate coverage supports it.
- When coverage is insufficient, explicitly say the limitation instead of summarizing confidently.
- Highlight new papers, representative papers, active trends, and likely missing directions as different things.

## Field guidance

- `analysis_mode`: copy the supplied analysis mode exactly.
- `topic_summary`: concise landscape summary grounded in the evidence actually provided.
- `branches_subthemes`: cluster the topic into named branches with short descriptions.
- `foundational_papers`: older or defining papers among the candidates only.
- `representative_papers`: papers that best represent major branches among the candidates.
- `recent_valuable_papers`: newer papers that appear especially relevant or informative from metadata, ideally covering multiple branches when possible.
- `lower_priority_or_overhyped`: use sparingly and cautiously; only when metadata suggests weak novelty, vague framing, or overclaiming.
- `recent_trends`: concrete trends visible in the candidates.
- `open_questions`: research questions implied by the scan.
- `evidence_quality_note`: explicitly mention candidate-count limits, provider failures/rate limits, metadata-only limits, or weak memory coverage.
- `missing_directions`: important directions likely absent or under-retrieved from the candidate list.
- `research_frontier`: where the field seems to be moving based on current evidence.
- `cross_paper_insights`: short grounded comparisons across multiple papers.
- `method_comparison_summary`: one grounded paragraph. In metadata-only mode, keep it conservative.
- `evolution_notes`: how this scan differs from prior topic understanding, or note that this is a first pass.

## Refinement pass

If `INITIAL_TOPIC_REPORT_JSON` is not `null`, this is a refinement pass.

- Do not repeat the initial draft.
- Improve paper selection balance, coverage discussion, research frontier, and missing directions.
- Keep the same schema and output one full replacement object.

## Topic

{{TOPIC}}

## Candidate papers

{{CANDIDATE_PAPERS_JSON}}

## Related paper memories

{{RELATED_PAPER_MEMORIES_JSON}}

## Prior claims summary

{{PRIOR_CLAIMS_SUMMARY_JSON}}

## Initial topic report JSON

{{INITIAL_TOPIC_REPORT_JSON}}
