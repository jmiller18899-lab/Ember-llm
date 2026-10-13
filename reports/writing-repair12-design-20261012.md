# WR12 semantic-token loss experiment — October 12, 2026

The user requested continuing Ember training after reviewing WR11. WR11 selected step 26 and reached 715/744 versus Repair2's 714/744. It fixed recipient perspective and one consent shortening case; benchmark shortening outputs, five development reminder omissions, and eleven force errors were unchanged. WR11 remains a private experimental candidate.

## Controlled training change

Start from frozen Repair2 `daf938bba5d4e6b650ec9d34a2d3ac56706cf549`, with the same Qwen base, router, graders, seed 431, learning rate 7.5e-7, 416 examples, one epoch, accumulation 8, and 52 steps. Save steps 13/26/39/52.

All WR11 training prompts, sources, preferred answers, rejected copies, rejected over-edits and shuffled order are deliberately reused (IDs change to WR12). This is a loss ablation, not a claim of new training data. The 256 retention and 64 recipient rows receive the same SFT loss as before.

For the 96 shortening rows, add `0.5 * mean CE(semantic answer tokens)` to the existing preferred CE, copy margin and over-edit margin. Selected answer phrases are `Please remember`, `Do not forget`, `No one may`, `only if`, `only after`, and standalone `should`, `must`, `may`, `might`, `can`, `will`, `until`, `before`, `after`, `please`. Real tokenizer character offsets select all overlapping tokens. Prompt tokens and the answer end token receive no extra loss. Shared factual tokens still receive the existing preferred-answer and margin losses.

This tests the hypothesis that averaging the original losses over many shared fact tokens dilutes the training signal on reminder and force words. WR11's result supports trying the hypothesis; it does not prove the cause.

## Development and selection

Evaluate 112 development rows: the already-observed WR11 set (56) plus 56 fresh vocabulary variants (40 shortening, 16 recipient). Fresh variants reuse structural patterns; they test vocabulary transfer, not unseen structures. Check new sources against observed history and keep all development rows excluded from training.

Include the unchanged parent (step 0) in checkpoint selection. A trained checkpoint must turn at least one baseline reminder/force failure into a v3 pass. It must retain every baseline passing shortening case, introduce no new per-case copies or reminder/force errors, and not increase aggregate copies, force failures or reminder omissions. Rank eligible checkpoints by total shortening passes, fewer copies, then earlier step. Baseline wins when no checkpoint meets these conditions. An unrelated consent or recipient improvement alone does not satisfy the WR12 target.

After selection, evaluate the frozen 744 observed benchmark. Overall acceptance retains at least 715 passes, at most 5/30 benchmark copies, and zero shortening or protected-family regressions. Record observed and fresh development results separately during review. v3 remains an evaluation checker; no runtime gate or grader edits. Reserved final holdouts remain unloaded. All changed outputs require review before any promotion decision.

## Launch and persistence

One L4 job with a 90-minute timeout. WR11 took about 25 minutes; extra development evaluation may make this run longer, roughly 25–40 minutes. Current L4 pricing is $0.80/hour, so the GPU timeout bounds hardware cost at $1.20. No automatic retry, deployment or promotion.

Private output: `Jmiller18899/ember-qwen3.5-4b-writing-repair12-20261012`. Persist training inputs, source hashes, model checks, intermediate adapters, selected adapter, metrics and raw evidence on the Hub.

The connector grants jobs/read-repos but no write-repos. Use the existing GitHub workflow secret `EMBER_HF_TOKEN` to verify private persistence and submit the job, as WR11 did; do not retrieve or change the secret. The isolated branch workflow must pass CPU contracts and real-tokenizer preflight before the exact launch commit marker enables GPU submission.

## Validation

Nine local contracts run: six failed against inherited WR11 behavior before implementation, then eight passed with the tensor test deferred to the remote environment. Remote CPU job `6acda6759496b09289e6482f` passed all 1,461 pytest tests, including the tensor-loss gradient test, in 103.82 seconds. Real-tokenizer preflight passed: 288 semantic answer tokens, maximum sequence length 220 under the 384 cap, 3,760 observed history rows checked, and no fresh exact-source or training benchmark-frame overlap. Exact staged code hashes and the preflight specification are saved in `reports/evidence/wr12-preparation-20261012/remote-validation.json`.
