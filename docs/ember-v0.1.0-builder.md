# Ember v0.1.0 corpus builder

The six builder requirements in `config/corpus_v0.1.0.json` are implemented in
`jobs/ember_corpus_build_v010.py` and `jobs/ember_corpus_sources_v010.py`.
This is CPU data preparation, including tokenizer training. It does not train
Ember's model, submit a GPU job, export a checkpoint, or change promotion gates.
The authoritative v0.0.7 ZIP is verified by SHA256 and remains unchanged.

## Running it

Install the corpus dependencies, then run a small check against the real sources:

```bash
python -m pip install 'datasets==5.0.1' 'datasketch>=2.0' 'huggingface_hub>=1.4' 'sentencepiece>=0.2'
python jobs/ember_corpus_build_v010.py --smoke --target-total-tokens 1000000 --output-dir /tmp/ember-v010-smoke
```

The small run uses the production 16,384-piece tokenizer size and all nine
categories. Its status is `SMOKE_PASS`, with `production_ready: false`.
Full fixed coverage thresholds remain visible in `envelope_copy_audit`; a small
sample can fail those coverage thresholds while passing its document checks.
It can never produce the full corpus's `PASS`.

For the complete corpus, use an empty output directory on a CPU machine with
at least 16 GB RAM and 20 GB free scratch:

```bash
python jobs/ember_corpus_build_v010.py --output-dir /data/ember-v010-corpus
```

The **Ember v0.1.0 Corpus CPU** GitHub workflow exposes both modes and uses the
existing `H_F2` secret. Its branch marker launches only a 1M-token smoke; full
mode requires manual dispatch and verifies runner capacity first. Neither mode
contains a model-training or GPU-submission step. Dataset revisions are resolved
before collection and recorded in every report.

## What changed

| Requirement | Implementation |
| --- | --- |
| SmolTalk instructions | Stream `all/train`, exclude `apigen-80k`, require ordinary user/assistant messages, retain subset provenance. |
| Synthetic envelope copying | Stream the generator and use case-sensitive exact hashes, preserving different literal values. External sources retain MinHash dedup. |
| Finite sources | Apply the 1.0x hard candidate minimum and optional 1.3x headroom to OASST and both Glaive categories. Actual token quotas still must pass. |
| Tool formatting | Emit bare assistant/tool boundaries, canonical argument JSON, and raw result content; malformed calls are excluded. |
| Memory use | Store candidate text, order keys, parent/license lookups, selected rows, and split assignments in SQLite. Read only the two required columns for the license join. Write output and provenance incrementally. The MinHash index remains in memory. |
| Final audit | Re-audit the selected synthetic slice, validate serialized calls, check real recovery/interpretation provenance, and recount the final files with the saved tokenizer. |

Chunks of the same source document, and OASST replies sharing a parent prompt,
stay in the same split. An additional SQL check rejects source groups appearing
in both splits. This prevents exact source-group leakage; it does not establish
that every concept or related repository is unseen in validation.

Collection and actual-token selection both require at least two independent
source groups per category. One long recovery trace must not consume a small
category's entire budget and leave no separate validation example. Whole-group
splits can therefore overshoot a small token target; measured totals and split
sizes are reported explicitly, and the full 250M–350M gate remains enforced.

## Token counts and evidence

Independent SentencePiece calls each add an initial prefix. Summing those calls
overstates the size of a concatenated corpus. The builder carries an EOT plus
newline boundary into each subsequent count and gives an initial prefix only to
the first document in each output file. Tests compare this streaming count with
encoding the entire final file in one call.

Source line endings are normalized before hashing or tokenization, matching
Python's text-file reader. A real-source run caught 28 tokens of drift from CRLF
terminal output in three OpenHands trajectories. The integration fixture now
includes both CR and CRLF and independently checks whole-file counts.

`corpus_stats.json` records actual tokens per category and split, final artifact
hashes, exact source revisions, tokenizer seed coverage, dedup counts, finite
source shortages, final synthetic audits, and the production-readiness flag.
`provenance.json` records each selected document's source group, split, hash,
license metadata, and actual token count. Failed builds write `status: FAIL`.
Existing output directories are never overwritten.

The synthetic leakage flag concerns reserved target values; exact frozen
semantic prompts are also filtered from external source documents. It is not a
claim that all benchmark vocabulary has been removed from general text. The
frozen semantic evaluator and the 90-case batteries are unchanged.

The focused tests include deliberately wrong JSON arguments, malformed calls,
duplicate documents, insufficient actual tokens despite adequate character
headroom, train/validation group overlap checks, and a complete nine-category
fixture build with a real saved/reloaded SentencePiece model. Fixture data and
its 512-piece test tokenizer validate the implementation; they are not evidence
that the 300M-token production corpus has been built.

## Verified live-source smoke

On September 9, 2026, commit `a354265191b2772c7c8a16b219525f93ea3e2b32`
passed the [1M-target CPU smoke](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34407024936)
using all nine real/synthetic source categories and the 16,384-piece tokenizer.
The [repository CPU workflow](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34407028933)
passed 506 repository tests and 22 authoritative-package tests. The smoke workflow
also passed its 38 focused builder/generator tests (a subset of the repository suite).

| Saved split | Documents | Actual Ember tokens |
| --- | ---: | ---: |
| Training | 1,477 | 996,744 |
| Validation | 21 | 65,982 |
| Total | 1,498 | 1,062,726 |

The downloaded artifact ZIP, each reported output hash, and every document's
provenance hash were independently verified. Encoding each entire saved file in
one SentencePiece call reproduced the reported counts exactly. All nine
categories appear in both splits, with zero exact duplicate documents and no
shared source groups.

The selected 380 synthetic documents pass their document audit. They do **not**
meet the unchanged full-corpus coverage requirements: that audit records 44
failures, including sparse tool/shape/value coverage. Long source groups also
make the small validation split exceed its target, with several categories
represented by just one validation document. This confirms pipeline operation;
it does not establish production dataset readiness or model quality. The full
300M-target build and its resource usage remain unverified.

Exact source revisions, artifact hashes, category totals, audit results, and the
independent verification are preserved in
[`reports/ember-v0.1.0-builder-smoke.json`](../reports/ember-v0.1.0-builder-smoke.json).
The raw GitHub Actions artifact expires on October 9, 2026.

Source API references: [SmolTalk schema and subsets](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
and [Hugging Face streaming](https://huggingface.co/docs/datasets/stream).
