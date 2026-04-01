# Paper memory extraction

You are a careful research assistant. Your task is to read the **paper excerpts** provided below (from the same paper) and produce a **single JSON object** that summarizes what the paper actually says.

## Rules (faithfulness)

1. **Only use information supported by the excerpts.** If something is unclear or not stated, say so briefly or use empty strings / empty arrays for that field.
2. **Do not invent citations, numbers, dataset names, or claims** that do not appear in the excerpts.
3. **Paraphrase** in your own words; do not copy long spans verbatim except short technical terms where needed.
4. If the excerpts are incomplete (e.g. missing references or appendices), reflect uncertainty in `limitations` or `open_questions` rather than guessing.
5. Output **valid JSON only** matching the required schema (no markdown fences, no commentary).
6. Prefer concrete technical descriptions over generic wording. Avoid phrases like "improves performance" without mechanism.

## Required JSON schema (field meanings)

- **title**: Paper title as given or inferred from the excerpts only.
- **problem**: The problem or gap the paper addresses.
- **core_idea**: The main idea or hypothesis in one or two sentences.
- **key_insight**: The most non-obvious idea or trick that makes this work.
- **method_overview**: High-level description of the proposed approach.
- **key_components**: Bullet-style list of main modules, stages, or components (short strings).
- **design_rationale**: Why each key component is needed and what problem it solves
- **training_inference_summary**: How training and/or inference work at a high level; use "Not specified in excerpts." if absent.
- **strongest_assumptions**: Explicit or implicit assumptions the method relies on.
- **comparison_perspective**: What type of methods this belongs to and what alternatives exist.
- **main_contributions**: What the paper claims to contribute (list of short strings).
- **why_it_works**: Mechanism-level explanation of why the method succeeds.
- **limitations**: Stated or reasonably implied limitations from the excerpts.
- **glossary**: Important terms with **term** and **definition** (definitions faithful to usage in the excerpts).
- **failure_modes**: Cases where the method likely breaks.
- **open_questions**: Questions left open, future work, or unclear points—not speculation beyond the text.

## Paper excerpts

{{PAPER_TEXT}}
