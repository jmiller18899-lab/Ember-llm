# Frozen tool-assistant confirmation: FAIL

The strongest observed router and extraction helper were frozen together and
tested on 100 new requests. The combined candidate did not pass its release
gate. Its model weights, fitted heads, extraction logic, and source lock were
preserved; no fixes were tuned against this confirmation.

| Measurement | Full precision | INT4 checkpoint |
| --- | ---: | ---: |
| Correct route | 87/100 | 90/100 |
| Exact tool arguments or correct arithmetic meaning | 62/80 | 65/80 |
| Combined success | 82/100 | 85/100 |
| Incorrect routes | 13 | 10 |
| Argument errors despite a correct route | 5 | 5 |

All 20 direct requests routed correctly; this does not mean Ember answered
them correctly. Direct-answer quality is evaluated separately. The INT4 model
was dequantized to float32, matching the original experiments.

## What failed

The unchanged helper drops arithmetic grouping: `18*(7+3)` became `18*7+3`,
and `(96-24)/6` became `96-24/6`. A negative leading number was not extracted.
Time requests sometimes retained unwanted words in the timezone value, such as
`Dushanbe, please` and `Praia show`.

The router also confused weather and time paraphrases and missed broader web
requests, including ferry timetables, application deadlines, and service
disruptions. Some web requests were routed as direct answers. These errors show
that the earlier narrow development results did not establish broad tool use.

## Evidence and packaging correction

The [first measurement](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34564981796)
at `63ef6ba4b6438913f1a95a3baa33fb9128bf5c89` produced the scores above.
Its default Bash log pipeline masked the evaluator's nonzero exit and allowed
an incorrectly named prototype artifact to be created. That was a workflow
defect, not a passing model result.

The [corrected gate run](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34565654365)
at `ec415b1d6b238a6c038e72ddaa64d77089df402e` reproduced the same scores,
failed the workflow, and skipped both packaging and prototype upload. Bash now
uses pipefail, and packaging independently verifies the measured report,
candidate manifest hash, both precision totals, and individual request records.
This rerun is reproduction of the same consumed suite, not a second independent
confirmation.

The [artifact retraction](https://github.com/jmiller18899-lab/Ember-llm/actions/runs/34565654303)
removed only artifact `10185693838`. Its evidence artifact `10185692097` was
preserved. The corrected run's evidence artifact is `10185934664`. The one-time
retraction workflow was then removed. There is no approved prototype artifact.

The full measured report is committed in
[`reports/ember-tool-assistant-confirmation.json`](../reports/ember-tool-assistant-confirmation.json).
It includes every request, expected result, actual call, margin, and failure.

## Next repair priorities

1. Preserve grouping and unary signs in arithmetic; reject unsupported syntax
   before producing a call. Use development tests for the grammar itself.
2. Tighten location extraction so it returns an unambiguous location or asks
   for clarification instead of attaching surrounding request text.
3. Broaden development coverage of direct/tool intent and weather/time/search
   distinctions, especially current-information requests beyond software releases.
4. Freeze any repaired candidate and author a different confirmation suite.
   Keep these first failures available as historical evidence.

The independent final-block direct-answer learning trial is described in
[`ember-direct-answer-results.md`](ember-direct-answer-results.md). It reached
only 2/24 task-aware checks and did not pass its learning gate. It cannot make
this failed tool candidate release-ready.
