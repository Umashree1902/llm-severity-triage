# Data Dictionary

Variables marked "derived" are computed during cleaning or produced by the LLM runs.

| Variable | Type / unit | Description |
|---|---|---|
| bug_id | Integer (identifier) | Unique Bugzilla bug number |
| creation_time | Timestamp (UTC) | Date and time the report was filed |
| product | Categorical | Top-level product (e.g., Firefox, Core) |
| component | Categorical | Second-level component within the product |
| type | Categorical | Bug type; only "defect" is retained |
| summary | Text (characters) | One-line title of the report |
| description | Text (characters) | First comment on the bug, i.e., the original report |
| platform / op_sys | Categorical | Hardware platform and operating system reported |
| keywords | Categorical list | Controlled tags such as "regression" |
| priority | Ordinal (P1–P5) | Triage priority; used only descriptively |
| status | Categorical | Workflow state at extraction (e.g., UNCONFIRMED, NEW, ASSIGNED, RESOLVED, VERIFIED) |
| resolution | Categorical | Outcome of a closed bug (e.g., FIXED, DUPLICATE, INCOMPLETE, WORKSFORME); empty if open. Used to identify duplicate pairs |
| final_severity | Ordinal (S1–S4) | Human-assigned severity at extraction; reference label |
| first_severity | Ordinal (S1–S4); derived | First non-default severity from the history |
| severity_changes | Count; derived | Number of changes to severity after first assignment |
| days_to_triage | Days; derived | Days from filing to first non-default severity |
| dupe_of | Integer (identifier) | Bug that this report duplicates, if any |
| model_id | Categorical; derived | LLM and version used for the run |
| prompt_condition | Categorical; derived | P0 (text only), P1 (with context), P2 (guardrailed) |
| run_id | Integer (1–5); derived | Repetition number for consistency analysis |
| llm_severity | Ordinal (S1–S4); derived | Severity returned by the LLM |
| llm_confidence | Proportion (0–1); derived | Confidence reported with the LLM label |
| deferred | Binary; derived | 1 if the guardrail deferred the defect to a human |
| signed_error | Integer (−3 to +3); derived | LLM ordinal score minus human ordinal score |

## Severity mapping

| Mozilla level | Study label | Ordinal score |
|---|---|---|
| S1 | Critical | 4 |
| S2 | High | 3 |
| S3 | Medium | 2 |
| S4 | Low | 1 |
