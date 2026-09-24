"""Frozen Ember runtime policy (promoted 2026-09-24): Repair2 weights + scoped rules + deterministic arithmetic tool.

respond(prompt, generate) is the exact composition evaluated in HF job via Actions run 35952086798
(jobs/ember_4b_repair2_arith_tool_eval.py): if the arithmetic tool fires, its answer is returned; otherwise
generate(system_prompt, prompt) is called with the promotion system prompt plus any scoped temporal/drafting rule.
Both source files are loaded by SHA-256 so the policy cannot drift from what was evaluated.
"""
import hashlib,types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODEL="Jmiller18899/ember-qwen3.5-4b-repair2"; MODEL_REV="daf938bba5d4e6b650ec9d34a2d3ac56706cf549"
BASE="Qwen/Qwen3.5-4B"; BASE_REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PINNED={
 "jobs/ember_arith_tool.py":"d4dd1d38f4e34d3f7f3262d2bb1d3bfff2aecfc6d5af23b02ae39fd46d31bac1",
 "jobs/ember_4b_repair2_original_promotion_eval.py":"48272629067857fc873dfab6c71c61e82c90eb4bf5cab7d85a7f5c9108b29ffc",
}
def _load(rel,name):
    data=(ROOT/rel).read_bytes(); got=hashlib.sha256(data).hexdigest()
    if got!=PINNED[rel]: raise RuntimeError(f"{rel}: sha256 {got} != frozen {PINNED[rel]}")
    m=types.ModuleType(name); exec(compile(data.decode(),rel,"exec"),m.__dict__); return m
TOOL=_load("jobs/ember_arith_tool.py","ember_arith_tool")
RULES=_load("jobs/ember_4b_repair2_original_promotion_eval.py","ember_scoped_rules")
system_for=RULES.system_for

def respond(prompt,generate):
    """Returns (answer, route). route is 'tool:<kind>' or 'model'."""
    hit=TOOL.solve(prompt)
    if hit: return hit[1],f"tool:{hit[0]}"
    return generate(system_for(prompt),prompt),"model"
