"""Ember runtime policy v2: Repair2 weights + scoped temporal/drafting rules + scoped context rule + arithmetic tool.

respond(prompt, generate) is the composition evaluated as "v2_context" in jobs/ember_4b_policy_v2_context_eval.py:
the arithmetic tool answers if it fires; otherwise generate(system_prompt, prompt) runs with the promotion system
prompt plus any scoped temporal/drafting rule, plus CONTEXT_RULE when the context gate fires. All three source files
are loaded by SHA-256. v1 (jobs/ember_runtime_policy.py) is left unchanged for rollback.
"""
import hashlib,types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL="Jmiller18899/ember-qwen3.5-4b-repair2"; MODEL_REV="daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PINNED={
 "jobs/ember_arith_tool.py":"d4dd1d38f4e34d3f7f3262d2bb1d3bfff2aecfc6d5af23b02ae39fd46d31bac1",
 "jobs/ember_4b_repair2_original_promotion_eval.py":"48272629067857fc873dfab6c71c61e82c90eb4bf5cab7d85a7f5c9108b29ffc",
 "jobs/ember_context_rule.py":"239a1d29e55be77d1f46867dc8cc6d94b82e39823b33f9ffd8ab1b56ffc8cb70",
}
def _load(rel,name):
    data=(ROOT/rel).read_bytes(); got=hashlib.sha256(data).hexdigest()
    if got!=PINNED[rel]: raise RuntimeError(f"{rel}: sha256 {got} != frozen {PINNED[rel]}")
    m=types.ModuleType(name); exec(compile(data.decode(),rel,"exec"),m.__dict__); return m
TOOL=_load("jobs/ember_arith_tool.py","ember_arith_tool")
RULES=_load("jobs/ember_4b_repair2_original_promotion_eval.py","ember_scoped_rules")
CONTEXT=_load("jobs/ember_context_rule.py","ember_context_rule")

def system_for(prompt):
    s=RULES.system_for(prompt)
    return s+" "+CONTEXT.CONTEXT_RULE if CONTEXT.context_gate(prompt) else s

def respond(prompt,generate):
    """Returns (answer, route). route is 'tool:<kind>' or 'model'."""
    hit=TOOL.solve(prompt)
    if hit: return hit[1],f"tool:{hit[0]}"
    return generate(system_for(prompt),prompt),"model"
