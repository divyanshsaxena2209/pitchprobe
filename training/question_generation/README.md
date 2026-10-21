# PitchProbe question-generation fine-tuning (experimental)

This is a **separate experiment** from the active fine-tuned LOW/MODERATE/HIGH classifier. It does not change the default targeted-question rules or production `master`.

## Why this experiment exists

In the Sonora seed-deck smoke test, multiple distinct claims received identical market-sizing, forecast, or conversion questions. The current generator is a hand-written topic router. The goal here is to learn claim-specific, evidence-seeking counter-questions from reviewed **claim → question pairs**, while preserving the risk label and topic in the prompt.

## Data requirements

Create `training/question_generation/claim_question_pairs.csv` using `claim_question_pairs_template.csv` as the schema. One row should represent one reviewed training example.

Required columns:

- `example_id`: unique stable ID.
- `claim`: exact sentence/claim the investor sees.
- `risk`: `Low`, `Moderate`, or `High`.
- `dimension`: e.g. Market, Traction, Pricing, Unit Economics, Team, Product, Technology, Privacy, Fundraising.
- `context`: optional short nearby evidence/context; never include personal or confidential data without permission.
- `question`: one specific question that probes the claim's weakest assumption or asks for evidence; it must end with `?`.
- `split`: fixed `train`, `validation`, or `test`.
- `source_group_id`: all examples from the same original source/deck/company must share a group ID.
- `source_name`, `source_url`, `license`, `attribution`: provenance and reuse terms.
- `reviewed`: must be `true` only after a human has checked that the question matches the claim and does not assert unsupported facts.

**Do not use the Sonora smoke-test deck or its generated PitchProbe questions as training labels.** Keep that deck as a private regression/evaluation example to reduce leakage and preserve an independent check.

The public [Startup Investor Falsification Question Bank](https://huggingface.co/datasets/mheilimo/startup-investor-falsification-question-bank) has 48 reusable questions across 12 diligence dimensions and lists CC BY 4.0. It is useful as an attributed source of question patterns, but it does **not** provide a matched claim plus target question for every item; it is not, by itself, a valid seq2seq training set. Attribute Playful Pixels Oy and track adaptations if you reuse its content. The [private-company due-diligence checklist](https://huggingface.co/datasets/mheilimo/private-company-due-diligence-checklist) lists CC BY 4.0 but only 12 checklist rows, also insufficient by itself for fine-tuning.

## Split and quality rules

- Aim initially for **at least 200–500 reviewed pairs**, with 10–15% validation and 10–15% held-out test, if feasible. This is a starting target, not a guarantee that the model will generalize.
- Split by `source_group_id`, never by row, so variants from one source cannot leak between splits.
- The trainer rejects rows not marked reviewed, invalid risks, duplicate IDs, empty fields, unsupported split names, or a source group appearing in multiple splits.
- Questions must be answerable in principle from evidence, specific to the claim, concise, and phrased as one question. Avoid generic "what evidence supports this?" unless no narrower probe is justified.
- Do not use a generated question from the model itself as a gold label without independent human review.

## Train and compare

From the repository root in a dedicated Python environment (Google Colab is fine):

```bash
pip install -r training/requirements.txt
python -m training.train_question_generator --data-csv training/question_generation/claim_question_pairs.csv --output-dir training/artifacts/pitchprobe-question-generator-v1
```

The script fine-tunes `google/flan-t5-small` by default, selects checkpoints on validation loss, and only then generates outputs for the held-out test set using both the base and fine-tuned generator. It writes side-by-side outputs for **human review**. It also calculates basic diagnostics such as question-mark rate, distinct-output rate, and exact-match rate; these are not substitutes for relevance, factuality, or usefulness review.

Do not enable this generator in PitchProbe just because training completes. Review a blinded sample and compare:
1. claim-specific relevance and whether the question targets a material assumption;
2. duplicate/generic question rate across different claims;
3. unsupported assumptions introduced by the generated question;
4. answerability from plausible evidence;
5. latency and memory use.

The existing targeted-rule generator stays the default until the fine-tuned generator wins a held-out human review and backend regression tests pass. The experimental branch intentionally does not wire the new checkpoint into the live request path.
