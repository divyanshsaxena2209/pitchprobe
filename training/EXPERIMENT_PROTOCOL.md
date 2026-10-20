# PitchProbe Fine-Tuning Lab

This branch, finetuning-lab, is an isolated research workspace based on the current PitchProbe master snapshot.

## Non-negotiable isolation rules

1. Do not make routine application, UI, or backend changes on this branch. Keep work inside training/ unless a training-only test requires otherwise.
2. Do not merge this branch into master just because training completes. Only propose integration after held-out evaluation and an explicit review.
3. Do not change the deployed/current model defaults or checkpoint paths while experimenting. Any fine-tuned checkpoint is an experiment artifact until evaluated.
4. Use external public datasets only. Do not use sonora_seed_pitch_deck.pdf, personal uploads, PitchProbe analysis outputs, generated predictions from PitchProbe, or private/company documents in training, validation, or test data.
5. Keep downloaded datasets, derived CSVs and model checkpoints local/Colab-only. The repository .gitignore excludes training/data/external_claims.csv and training/artifacts/. Never force-add them.
6. Preserve dataset source, version, URL, license, attribution and transformations in experiment notes. Check the original dataset card before each new download or redistribution.

## Experiment phases

### Phase A — Classification baseline
- Source: FinClaimRadar on Hugging Face.
- Its source labels describe evidentiary-basis categories, not PitchProbe's exact risk taxonomy.
- The mapping to LOW, MODERATE, HIGH is an explicitly approximate research proxy, not gold truth for startup pitches.
- Run baseline evaluation and retain the untouched baseline report.

### Phase B — Classifier fine-tuning
- Fine-tune DeBERTa only on the training split.
- Use validation data for checkpoint selection and early stopping.
- Evaluate once on the held-out test split.
- Compare per-class precision, recall and F1, macro-F1, and the confusion matrix against the baseline.
- Do not deploy based solely on training loss or a single aggregate score.

### Phase C — Pairwise exaggeration research

- Source: [Scientific Exaggeration Detection](https://huggingface.co/datasets/copenlu/scientific-exaggeration-detection), labeled GPL-3.0 by its dataset card.
- Task: given both a press-release conclusion and its scientific-abstract conclusion, predict `same`, `exaggerates`, or `downplays`.
- Use the official 100-row train split only for fitting and grouped internal validation; reserve the official 563-row test split for final evaluation only.
- Keep the task-specific checkpoint separate from the FinClaimRadar risk classifier. This experiment measures scientific/health press-release exaggeration and is not startup-pitch risk validation.
- Keep a licence and attribution record and review GPL-3.0 obligations before redistributing derived data or artifacts.

### Phase C — Question-generation research
- Keep the application's targeted-question fallback unchanged.
- Treat the Startup Investor Falsification Question Bank (CC BY 4.0) as an external question source, not a ready-made claim/question training set.
- Build a separate, documented matched-pair dataset only from appropriately licensed public materials and reviewed source-to-question matches.
- Do not fine-tune FLAN-T5 on unrelated financial QA pairs or on its own unreviewed outputs.

### Phase D — Integration proposal
Only after evaluation, create a separate pull request proposing integration. Include the experiment configuration, exact dataset revision and license, label mapping, split method, metrics, errors, limitations and checkpoint location. Master remains unchanged unless that proposal is explicitly accepted.

## External sources recorded for this experiment

- FinClaimRadar: https://huggingface.co/datasets/ImBinarry/FinClaimRadar-Financial-Reasoning (CC BY-SA 4.0 according to its dataset card; keep attribution and check share-alike obligations).
- Startup Investor Falsification Question Bank: https://huggingface.co/datasets/mheilimo/startup-investor-falsification-question-bank (CC BY 4.0; attribute Playful Pixels Oy, link the license, and indicate changes).
- FinQA: https://huggingface.co/datasets/ibm-research/finqa (financial-report QA; auxiliary only, not investor-objection ground truth).

A model that improves on these external proxy labels is not thereby validated on real startup pitch decks. Report that limitation whenever sharing results.

### Observed Phase C comparison (Colab, 2026-10-21)

On the official 563-pair test split, the published [`copenlu/scientific-exaggeration-detection`](https://huggingface.co/copenlu/scientific-exaggeration-detection) strength model was run in inference-only mode on each press-release and abstract conclusion. The sign of the difference between its predicted four-level strengths generated the final paired label.

| Approach | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority baseline (`same`) | 0.6128 | 0.2533 |
| Fine-tuned DeBERTa pair classifier | 0.5542 | 0.2875 |
| Published claim-strength model + gap rule | **0.6785** | **0.6217** |

For the published strength model, class F1 was 0.761 for `same`, 0.614 for `exaggerates`, and 0.490 for `downplays`. Confusion matrix (rows actual; columns predicted; order `same`, `exaggerates`, `downplays`): `[[252,49,44],[30,81,11],[35,12,49]]`.

Interpret this as a strong in-domain reference baseline, not startup-pitch validation. The paper's repository states that its paired benchmark test file was used in the paper's experiments, so this is an in-domain reproduction/reference result rather than an independent blind evaluation of the published model. The released model was designed for causal claim-strength classification and comparison on scientific press-release/abstract pairs; this result does not establish generalization to startup claims. Keep it isolated from the app and master branch. Preserve the local Colab artifacts and review the model/dataset licensing terms before any redistribution.

### Phase D — Human-reviewed startup-risk labels and question retrieval (research-only)

- Current source review did not surface a sufficiently sized human-labelled startup-pitch dataset that directly matches PitchProbe's LOW / MODERATE / HIGH definitions.
- Do not promote mapped FinClaimRadar labels to gold risk labels; they identify evidentiary categories and are only a proxy.
- Use training/create_risk_review_sheet.py to sample training-split financial claims into an annotation worksheet. Complete human_risk_label, claim_is_investor_relevant, and evidence_status independently, and use second review/adjudication for disagreement. NEEDS_CONTEXT is an abstention state, not a fourth model class. This worksheet is a pilot annotation exercise, not a validated startup dataset.
- Use training/question_bank_router.py as a no-training retrieval baseline over the 48-question [Startup Investor Falsification Question Bank](https://huggingface.co/datasets/mheilimo/startup-investor-falsification-question-bank), pinned at f39e258288f16c6f85dc40e67d38b9e90eff1426. Dataset card lists CC BY 4.0; attribute Playful Pixels Oy and link the license. Prototype results require human review of contextual relevance before use in the app.
- Keep the application and master unchanged. Only propose integration after a test with reviewer-annotated startup claims and judged question quality.

