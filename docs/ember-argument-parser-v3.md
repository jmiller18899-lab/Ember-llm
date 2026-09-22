# Ember arithmetic and location parser v3

The separate `argument-parser-v3` revision passed **60/60 newly authored parser
requests**, with the tool choice explicitly supplied. It repairs the five known
argument failures and asks for clarification on unsupported or ambiguous input.
It is available through an opt-in runtime and a model-free parsing CLI.

The original v2 candidate, model weights, routing heads, source lock, and failed
100-request report are preserved. V3 does not replace the default runtime or
qualify the combined tool assistant for release.

## Measured behavior

| Parser-only check | Passed | What counted |
| --- | ---: | --- |
| Arithmetic calls | 20/20 | Exact expression, independently specified numeric result, one fixture call |
| Weather/time calls | 20/20 | Exact location or time zone, one fixture call |
| Clarifications | 20/20 | Explicit reason and message, no call, no fixture dispatch |

The parser, runtime adapter, evaluator, and development tests were hashed before
these requests were written. The suite is disjoint by normalized request text
from the recorded historical hashes, router training, original 100-request
confirmation, and literal parser development examples. It is assistant-authored
grammar coverage, not independent human evaluation or a representative traffic
sample. Repeating it is regression evidence, not another fresh confirmation.

The [source lock](../tool_assistant/data/parser-v3-source-lock.json),
[60 requests and expected outcomes](../tool_assistant/data/parser-v3-confirmation.json),
and [complete first report](../reports/ember-parser-v3-confirmation.json) preserve
the inputs, outputs, hashes, and all fixture calls. The evaluator refuses to
overwrite a report and exits unsuccessfully if any case fails.

The earlier failures are development regressions, separate from that result:

| Known request | V2 argument | V3 argument |
| --- | --- | --- |
| `18*(7+3)` | `18*7+3` | `18*(7+3)` = 180 |
| `(96-24)/6` | `96-24/6` | `(96-24)/6` = 12 |
| `-17 plus 42` | Missing | `-17+42` = 25 |
| Current time in Dushanbe, please | `Dushanbe, please` | `Dushanbe` |
| What would a clock in Praia show at this moment? | `Praia show` | `Praia` |

## Arithmetic contract

V3 validates the whole expression before removing whitespace. Parentheses,
unary signs, decimals, scientific notation, and standard operator precedence
are preserved. It recognizes a limited set of request frames and arithmetic
words, including percent-of, squared/cubed, and integer powers. Well-formed
thousands separators and Unicode minus/multiply/divide signs are supported.

English `-3 squared` means `(-3)**2`; symbolic `-3**2` retains conventional
precedence and means `-(3**2)`. Consecutive English postfix powers require
explicit grouping. Operators are `+`, `-`, `*`, `/`, and `**`. Factorials,
remainder, floor division, implicit multiplication, units, functions, names,
extra clauses, and malformed number groups produce clarification. In
particular, `5!` must not be reduced to `5`, or `1 2` to `12`.

An allowlist and a bounded AST walker validate syntax and intermediate values;
there is no `eval` or `exec`. Validation uses exact rational arithmetic. Limits
are 200 expression characters, 60 AST nodes, magnitude at most `10**15`, and
integer power/scientific exponents from -12 through 12. Division by zero and
zero to a negative power produce no call. The calculator handler still owns
execution and its numeric precision; parsing does not round the expression.

## Location and time contract

V3 recognizes `in`/`for` location slots and explicit `location:`/`timezone:`
fields. It removes only known complete polite/current-time suffixes and the
trailing verb in a clock question. Unicode letters, internal apostrophes,
commas, ordinary geographical abbreviations, and hyphenated names are retained.
IANA time zones are validated with `zoneinfo`; they are accepted for the clock
tool, not as weather locations.

Multiple places, relative places such as “my city”, recognized future/historical
modifiers, unknown IANA zones, and detected trailing clauses require
clarification. Names containing conjunctions or otherwise reserved
words can be supplied as a single quoted name, for example
`Weather in "Trinidad and Tobago" now?`. Quoting identifies the name boundary;
it does not establish that a place exists.

This is parsing, not geocoding or intent classification. A live adapter must
still resolve a city to a unique place/time zone and ask for a region when
needed. Unusual valid names may require quotation or clarification; a
syntactically plausible name can still be nonexistent. Routing errors remain
possible because the router is unchanged.

## Try the parser

From this repository checkout, Python's standard library is sufficient:

```sh
python -m tool_assistant.resolver_v3 calculator 'Calculate 18*(7+3).'
python -m tool_assistant.resolver_v3 get_time 'Time in Dushanbe, please?'
python -m tool_assistant.resolver_v3 weather 'Weather in Paris and Rome?'
```

Results retain the tool/key/value/payload shape and add `reason` and
`clarification`. A failed parse returns `payload: null`.

With a locally built frozen candidate and the repository's CPU dependencies:

```python
from tool_assistant.runtime_v3 import ParserRuntime

runtime = ParserRuntime("frozen-candidate", precision="int4")
proposal = runtime.plan("Calculate 18*(7+3).")
```

`ParserRuntime` inherits the frozen router and explicit handler dispatch. Its
`plan` method uses v3 parsing. `needs_clarification` prevents dispatch;
`direct_answer_unavailable` remains explicit. The original `Runtime`, build,
100-request evaluation, and prototype packaging continue to use v2.

## Reproduce validation

```sh
python -m pytest -q tests/test_resolver_v3.py tests/test_parser_evaluation.py tests/test_tool_assistant.py
python -m tool_assistant.evaluate_parser --report /tmp/parser-v3-reproduction.json
```

All **168 focused tests** passed before freezing. These cover meaning and
grouping, malformed inputs, place boundaries, all 48 original argument-bearing
training examples, suppression of invalid dispatches, and evaluator failure
detection. Repository CPU validation also runs the parser suite and reproduces
the 60-request result. The full repository suite passed **660/660 locally**
after installing its CI-pinned `jsonschema==4.23.0` dependency. No model or live
service is called by the parser check.

The next combined candidate needs routing improvements, a new source lock that
includes v3, and a different full-model confirmation. The existing 82/100 full
precision and 85/100 INT4 combined scores remain historical failure evidence.
