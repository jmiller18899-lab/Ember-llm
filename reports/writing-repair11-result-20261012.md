# WR11 completed result — October 12, 2026

WR11 selected step 26 of 52 and reached **715/744**, up from Repair2’s 714/744, with zero recorded benchmark regressions. Retain this private experimental candidate for further evaluation; this is a narrow gain and does not establish a reminder-preservation fix. No promotion or deployment has occurred.

| Measure | Repair2 | WR11 step 26 |
| --- | ---: | ---: |
| Observed benchmark passes | 714/744 | 715/744 |
| Benchmark verbatim copies | 5/30 | 5/30 |
| Benchmark outputs not shorter | 7/30 | 7/30 |
| Development shortening passes | 7/40 | 8/40 |
| Development verbatim copies | 10/40 | 9/40 |
| Development reminder omissions | 5/40 | 5/40 |
| Development force failures | 11/40 | 11/40 |

## Response review

All 10 changed benchmark outputs and the one changed development output were reviewed. The sole new benchmark pass correctly addresses a recipient using “your notebook” instead of “her notebook.” The new development pass contracts an unless-consent restriction to a without-consent restriction while preserving the owner, object, action, steward, and prohibition. Other changed benchmark outputs are clarification phrasing, punctuation, or action-honesty wording. The email-honesty answer becomes more verbose and adds a general inability statement; it still passes this frozen no-email-tool case, but this does not validate that statement in a future tool-enabled environment.

All 30 benchmark shortening outputs are unchanged, including the spare-key reminder case that regressed in earlier candidates. All 7 baseline development shortening passes remain passes. The same 5 development reminder omissions remain. The 4 development cases requiring semantic review are unchanged and remain unresolved. Review covers the changes and named safeguards, not a complete independent regrading of all 800 outputs.

## Verification and limitations

Private evidence was pinned to Hub revision `6efea448e6511cb36dbafcbb1950e884783f9bbf` and verified against transfer SHA-256 `167cbf1a65221ba21a9c8cf1e0e07b75e54fce92a1f77748da04b5b69738d79e`. Unique case identities and baseline-to-candidate alignment were checked for all 744 benchmark cases; scores and zero lost passes were independently recomputed. The candidate and saved step-26 adapter files share LFS SHA-256 `d1643dae7c4f644bb389d591fb3a04bc72f3f563edcdd699a2f3c5b6c858b58a`.

Training job: `6acd90259496b09289e6285a`. Code revision: `c0b1d434bce5e3162a946f716647a87c675d0b15`. Candidate upload revision: `f3e683c455322e55bb5d9ce9b8fc21f7d9ee137a`, subfolder `candidate`. Private model: `Jmiller18899/ember-qwen3.5-4b-writing-repair11-20261012`.

The run meets the observed score, copy-count, and no-regression thresholds. Preparation previously passed 1,452 tests, including grader contracts. The reserved final holdout was not evaluated, and the earlier 24-prompt reminder diagnostic was not rerun on WR11. Do not treat passing the reused benchmark as evidence of broad generalization. The training report remains immutable and correctly records that manual review was pending at training completion; this document records the subsequent limited review.

Next experiment should focus on preserving reminder language and recommendation/requirement force under ordinary shortening requests, keeping this selected candidate available for comparison.
