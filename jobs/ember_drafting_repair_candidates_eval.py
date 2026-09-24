# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0"]
# ///
"""Experimental drafting-only additions; never deployed automatically."""
import re

_NAME = r"[^\W\d_][\w'’\-]*"
_MESSAGE = re.compile(
    rf"^\s*(?:please\s+)?(?:"
    rf"(?:text|tell|remind|ping)\s+{_NAME}\s+(?:that|about|to)\b|"
    rf"let\s+{_NAME}\s+know\b|"
    rf"message\s+{_NAME}\s*[:;,]|"
    rf"(?:write|send|draft)\s+(?:a\s+)?(?:quick\s+|short\s+)?"
    rf"(?:note|text|message|reply)\s+(?:to|for)\s+{_NAME}\b)", re.I)
_SHORTEN = re.compile(
    r"^\s*(?:(?:please|could you|can you)\s+)?"
    r"(?:shorten|condense|make\s+(?:this|it)\s+shorter)\b", re.I)

PERSPECTIVE_RULE = (
    "Recipient perspective: write the draft as the sender speaking directly to the named recipient. "
    "For belongings owned by that recipient, his, her, or their becomes your. "
    "Resolve ownership first: references to a different named person must remain about that other person, "
    "not become your. Keep the sender's I/my unchanged. Preserve the object, number, timing, negations, "
    "and uncertainty. Return only the draft; do not claim to have sent it."
)
SHORTENING_RULE = (
    "Concise rewrite: return only a genuinely shorter version of the supplied text. "
    "Remove redundant framing and unnecessary words rather than copying the original. "
    "Keep every supplied name, quantity, place, date, deadline, negation, condition, and degree of obligation. "
    "Do not turn a suggestion into a requirement or permission into a command. "
    "Do not add an introduction, quotation marks, new facts, or a claim that an action was performed."
)

def perspective_gate(prompt: str) -> bool:
    return bool(_MESSAGE.search(prompt))

def shortening_gate(prompt: str) -> bool:
    if not _SHORTEN.search(prompt):
        return False
    if ':' in prompt:
        return bool(prompt.split(':', 1)[1].strip(' \t\r\n\"\'“”‘’'))
    return bool(re.search(r'[\"“\']\s*\S.+?[\"”\']', prompt))

def augment(prompt: str, system: str, perspective: bool = False,
            shortening: bool = False, missing: bool = False) -> str:
    if missing:
        return system
    rules = []
    if perspective and perspective_gate(prompt):
        rules.append(PERSPECTIVE_RULE)
    if shortening and shortening_gate(prompt):
        rules.append(SHORTENING_RULE)
    return system + ((' ' + ' '.join(rules)) if rules else '')

import argparse, hashlib, json, os, types, urllib.request
from pathlib import Path

SOURCE_COMMIT = "dac42aaf7e743ae44c75891d96c753105ffd17fe"
FROZEN_EVAL_URL = ("https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/"
                   + SOURCE_COMMIT + "/jobs/ember_policy_v3_candidates_eval.py")
CONFIGS = {"v3_baseline": (False, False), "recipient_only": (True, False),
           "shortening_only": (False, True), "combined": (True, True)}
EXPECTED = {"exact72": 72, "temporal8": 8, "drafting8": 7, "promo_v2": 200,
            "heldout": 180, "ctx_holdout": 24, "suite_v3": 185, "probes": 28}

def norm(text):
    return re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()

def fresh_cases():
    """Declared before any candidate inference; development suites are not held-out."""
    rows=[]
    examples=[("Imani","her","coat","Sunday"),("Mateo","his","tablet","Thursday"),
              ("Seren","their","watch","Tuesday"),("Zora","her","mug","Saturday"),
              ("Ravi","his","compass","Wednesday"),("Lior","their","atlas","Monday"),
              ("Maren","her","helmet","Friday"),("Niko","his","tripod","Sunday"),
              ("Asha","her","wallet","Thursday"),("Dorian","his","flashlight","Tuesday"),
              ("Elowen","their","sweater","Saturday"),("Bryn","their","thermos","Monday")]
    templates=["Text {n} that I have {p} {obj} and can return it {d}.",
               "Tell {n} that I found {p} {obj} and can bring it back {d}.",
               "Message {n}: I picked up {p} {obj} and can return it {d}."]
    for i,(n,p,obj,d) in enumerate(examples):
        rows.append({"id":f"fresh-recipient-{i:02}","family":"recipient",
            "prompt":templates[i%3].format(n=n,p=p,obj=obj,d=d),
            "require":[n,obj,d],"object":obj,"kind":"recipient","scoring":"rubric"})
    for i,(n,owner,p,obj,place) in enumerate([
        ("Tamsin","Orin","his","camera","studio"),("Vera","Adira","her","scarf","foyer"),
        ("Jalen","Remy","their","badge","workshop"),("Suri","Bastian","his","coat","office"),
        ("Otis","Anouk","her","notebook","library"),("Freya","Ellis","their","umbrella","cafe")]):
        rows.append({"id":f"fresh-thirdparty-{i:02}","family":"thirdparty",
            "prompt":f"Text {n} that {owner} left {p} {obj} in the {place}.",
            "require":[n,owner,obj,place],"object":obj,"kind":"thirdparty","scoring":"rubric"})
    shortened=[
        ("Please make sure that all technicians meet at the east loading bay before 10:20 AM on Wednesday.", ["technicians","east loading bay","before 10:20","Wednesday"], []),
        ("Visitors must not enter the north gallery before noon on Friday.", ["visitors","north gallery","before noon","Friday"], ["not","never"]),
        ("All students are kindly asked to leave exactly three folders on the blue desk by 4 PM.", ["students","three folders","blue desk","4 PM"], []),
        ("Please remember that the spare key must remain in locker 27 until Monday.", ["spare key","locker 27","until Monday"], []),
        ("We would appreciate it if all drivers could wait at the west entrance after 7:45 PM on Thursday.", ["drivers","west entrance","after 7:45","Thursday"], []),
        ("It is important that nobody move the red crates from storage room 12 before Saturday.", ["red crates","storage room 12","before Saturday"], ["nobody","no one","not","don't","never"])]
    for i,(text,facts,neg) in enumerate(shortened):
        rows.append({"id":f"fresh-shorten-{i:02}","family":"shortening",
            "prompt":f"Shorten this while preserving every fact: {text}",
            "require":facts,"negative_any":neg,"source":text,"kind":"shortening","scoring":"rubric"})
    return rows

def legacy_fresh_score(row, output):
    n=norm(output)
    if not all(norm(x) in n for x in row["require"]): return False
    if re.search(r"\b(?:i have sent|i sent|message sent|i'll tell|i will tell|i'll message)\b", n): return False
    if row["kind"] == "recipient":
        return bool(re.search(r"\byour\s+"+re.escape(row["object"])+r"\b",n)) and not n.startswith(("text ","tell ","message "))
    if row["kind"] == "thirdparty":
        return not re.search(r"\byour\s+"+re.escape(row["object"])+r"\b",n) and not n.startswith(("text ","tell ","message "))
    return len(output.strip()) < len(row["source"]) and (not row["negative_any"] or any(x in n for x in row["negative_any"]))

# Grader v2 is deliberately bounded: it certifies conservative shortening edits,
# rejects known meaning changes, and sends unrecognized paraphrases to review.
# It is NOT a general semantic-equivalence model. Keep the historical checker
# available so reproducing old scores does not depend on the new rubric.
GRADER_VERSION = "meaning-preservation-v2"
_DAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_FORCE_PATTERNS = [
    ("no_obligation", r"\b(?:(?:are|is) not required to|not required to|do not have to|don't have to|need not|not necessary to)\b"),
    ("prohibition", r"\b(?:(?:must|shall) not|mustn't|(?:are|is) not allowed to|not allowed to|(?:are|is) forbidden to|forbidden to|do not|don't)\b"),
    ("prohibition", r"\b(?:nobody|no one)\b"),
    ("requirement", r"\b(?:(?:are|is) required to|required to|must|shall|have to|has to|(?:is|are) mandatory|mandatory|obligatory|require)\b"),
    ("recommendation", r"\b(?:should|ought to|(?:is|are) recommended|recommended|advised to)\b"),
    ("optional", r"\b(?:(?:is|are) optional|optional|voluntary)\b"),
    ("permission", r"\b(?:(?:are|is) allowed to|allowed to|(?:are|is) permitted to|permitted to|can)\b"),
    # 'may' is ambiguous between possibility and permission: do not conflate it
    # with 'can' or 'might', and do not silently certify such paraphrases.
    ("may", r"\bmay\b"),
    ("uncertainty", r"\b(?:might|could|possibly|perhaps|probably|maybe|likely|unlikely)\b"),
    ("request", r"\b(?:(?:are|is) (?:kindly )?asked to|(?:kindly )?asked to|please|kindly)\b"),
]
_FORCE_RE = re.compile("|".join(f"(?P<f{i}>{p})" for i,(_,p) in enumerate(_FORCE_PATTERNS)))
_TOKEN_RE = re.compile(r"[+-]?\d+(?:[.:]\d+)*|[^\W\d_]+(?:'[^\W\d_]+)?|__negative_subject__|[$€£¥%/+=<>-]", re.UNICODE)


def _meaning_view(text):
    """Extract a conservative, order-sensitive signature; no bag-of-words pass."""
    text = norm(text).replace("−", "-")
    text = re.sub(r"^please (?:remember|note)(?: that)?\s+", "", text)
    text = re.sub(r"^it is important that\s+", "", text)
    text = re.sub(r"\bplease (?:make sure(?: that)?|ensure)\s+", "please ", text)
    text = re.sub(r"^we would appreciate it if (.+?) (?:could|would)\s+", r"please \1 ", text)
    forces = []
    def remove_force(match):
        kind = _FORCE_PATTERNS[int(match.lastgroup[1:])][0]
        forces.append(kind)
        if kind == "uncertainty":
            return " " + match.group() + " "  # likely != unlikely; might != probably
        return " __negative_subject__ " if match.group() in ("nobody", "no one") else " "
    body = _FORCE_RE.sub(remove_force, text)
    # Courtesy does not weaken an explicit requirement: "Please, staff must...".
    distinct = set(forces)
    if len(distinct) > 1 and "request" in distinct and not re.search(r"\basked to\b", text):
        distinct.remove("request")
    force = next(iter(distinct)) if len(distinct) == 1 else ("statement" if not distinct else "mixed")
    negation = len(re.findall(r"\b(?:not|never|neither|without)\b", body))
    qualifiers = re.findall(r"\b(?:exactly|at least|at most|only|all|each|every)\b", body)
    conditions = re.findall(r"\b(?:only if|if|unless|provided that|as long as)\b", body)
    timing = re.findall(r"\b(?:no later than|before|after|until|ahead of)\b", body)
    numbers = re.findall(r"(?<![\w.])[+-]?\d+(?:[.:]\d+)*(?![\w.])", body)
    body = re.sub(rf"\bon (?={_DAYS}\b)", "", body)
    tokens = [t for t in _TOKEN_RE.findall(body) if t not in {"a", "an", "the", "that"}]
    return {"force": force, "negation": negation, "qualifiers": qualifiers,
            "conditions": conditions, "timing": timing, "numbers": numbers,
            "tokens": tokens}


def grade_case(row, output, legacy_scorer):
    """Apply the same versioned shortening rubric in EVERY evaluation suite.

    A review result is not a pass. Non-shortening tasks retain their old grade.
    Historical scores are returned separately, never relabelled as corrected.
    """
    if not isinstance(output, str) or not output.strip():
        return {"passed": False, "legacy_passed": False, "status": "fail",
                "reasons": ["empty_or_invalid_output"], "grader_version": GRADER_VERSION}
    legacy = bool(legacy_scorer(row, output))
    result = {"passed": legacy, "legacy_passed": legacy,
              "status": "pass" if legacy else "fail",
              "reasons": [] if legacy else ["legacy_requirements_failed"],
              "grader_version": GRADER_VERSION}
    if row.get("scoring") == "exact" or row.get("kind") not in ("shorten", "shortening"):
        return result
    source = row.get("source")
    if not isinstance(source, str) or not source.strip():
        return {**result, "passed": False, "status": "fail", "reasons": ["missing_source"]}
    before, after = _meaning_view(source), _meaning_view(output)
    reasons = list(result["reasons"])
    if len(norm(output)) >= len(norm(source)):
        reasons.append("not_shorter")
    if before["force"] != after["force"] and "mixed" not in (before["force"], after["force"]):
        reasons.append("force_changed:" + before["force"] + "->" + after["force"])
    for key in ("negation", "qualifiers", "conditions", "timing", "numbers"):
        if before[key] != after[key]:
            reasons.append(key + "_changed")
    if reasons:
        return {**result, "passed": False, "status": "fail", "reasons": reasons}
    # Unrecognized synonym, changed subject/action, or mixed modal scope is NOT
    # proof of equivalence. A human can accept it later; the counter cannot.
    if before["tokens"] != after["tokens"] or "mixed" in (before["force"], after["force"]):
        return {**result, "passed": False, "status": "review",
                "reasons": ["unverified_paraphrase_or_scope"]}
    return result


def fresh_score(row, output):
    return grade_case(row, output, legacy_fresh_score)["passed"]


def score_comparison(results, outputs, legacy_results, audit, leakage):
    """Compare corrected scores with a baseline graded by the SAME rubric."""
    reproduced = {s:legacy_results["v3_baseline"][s]["score"][0] == n for s,n in EXPECTED.items()}
    baseline = results["v3_baseline"]
    verdict = {}
    for config in results:
        if config == "v3_baseline":
            continue
        candidate = results[config]
        regressions = [{"suite":k[0], "id":v["id"]} for k,v in outputs["v3_baseline"].items()
                       if v["ok"] and not outputs[config][k]["ok"]]
        checks = {
            "legacy_baseline_reproduced": all(reproduced.values()),
            "no_individual_regressions": not regressions,
            "no_family_lower": all(v[0] >= baseline[s]["families"][f][0]
                for s in candidate for f,v in candidate[s]["families"].items()),
            "suite_v3_above_corrected_baseline": candidate["suite_v3"]["score"][0] > baseline["suite_v3"]["score"][0],
            "drafting_above_corrected_baseline": candidate["suite_v3"]["families"]["drafting"][0] > baseline["suite_v3"]["families"]["drafting"][0],
            "no_scope_leakage": not leakage,
            "frozen_audit_passed": not any(audit.values()),
        }
        verdict[config] = {"eligible_for_manual_review":all(checks.values()),
                          "failed_checks":[k for k,v in checks.items() if not v], "regressions":regressions}
    return reproduced, verdict


def rescore_records(records):
    """CPU-only replay of archived rows; never load a model or assume coverage."""
    rescored = []
    seen = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("legacy_ok"), bool):
            raise ValueError("Each record needs an explicit boolean legacy_ok")
        key = (record["config"], record["suite"], record["row"]["id"])
        if key in seen:
            raise ValueError("Duplicate evaluation record: " + repr(key))
        seen.add(key)
        grading = grade_case(record["row"], record["output"], lambda r,o:record["legacy_ok"])
        rescored.append({**record, "grading":grading, "ok":grading["passed"]})
    return {"grader_version":GRADER_VERSION, "records":rescored,
            "coverage":"supplied records only; not a full benchmark unless independently complete",
            "model_inference_performed":False, "automatic_promotion":False}


def build_route(E,M,config):
    original=E.policy(M,set(E.COMPONENTS))
    perspective,shortening=CONFIGS[config]
    def route(prompt):
        kind,value=original(prompt)
        if kind != "model" or not M.R.drafting_gate_v3(prompt,M.O.drafting_gate): return kind,value
        return kind,augment(prompt,value,perspective,shortening,missing=M.R.missing_text_gate(prompt))
    return route

def load_frozen():
    source=urllib.request.urlopen(FROZEN_EVAL_URL,timeout=60).read()
    E=types.ModuleType("frozen_v3")
    E.__file__="/tmp/ember-writing-isolated/jobs/frozen_v3.py"
    exec(compile(source.decode(),E.__file__,"exec"),E.__dict__)
    return E,E.load(),hashlib.sha256(source).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true")
    ap.add_argument("--rescore",type=Path,help="CPU-only regrade of a saved records JSON file")
    args=ap.parse_args()
    if args.rescore:
        payload=json.loads(args.rescore.read_text())
        records=payload if isinstance(payload,list) else payload["records"]
        print(json.dumps(rescore_records(records),indent=2)); return
    E,M,source_sha=load_frozen()
    routes={c:build_route(E,M,c) for c in CONFIGS}
    fresh=fresh_cases(); assert len(fresh)==24
    audit=E.audit(M,[])
    if audit["tool3_wrong_or_on_rubric"]: raise RuntimeError(audit)
    guard_prompts=["Calculate 13 + 8 - 4. Return only the number.",
        "owner=Rin; item=book; pickup=Friday. Whose item is it and when can it be collected?",
        "Translate this into French.","Shorten this.","My bus left at 6 AM. When did it arrive? Context: tonight.",
        "Copy exactly: Text Zora that her bag is ready.","Explain how to shorten a sentence."]
    for p in guard_prompts:
        assert all(route(p)==routes["v3_baseline"](p) for route in routes.values()),p
    print("WRITING_PREFLIGHT_PASS",json.dumps({"configs":list(CONFIGS),"fresh_cases":len(fresh),
        "source_commit":SOURCE_COMMIT,"source_sha256":source_sha,"weights_changed":False}),flush=True)
    if args.preflight: return

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer,Qwen3_5ForCausalLM
    from peft import PeftModel
    if not torch.cuda.is_available(): raise RuntimeError("GPU required; refusing CPU inference")
    bench=json.loads(Path(hf_hub_download(E.BENCH,"candidate.json",revision=E.BENCH_REV)).read_text())
    exact=[r for r in bench if r.get("scoring")=="exact"]; assert len(exact)==72
    suites=E.suites(M,exact)
    known_prompts={r["prompt"] for items in suites.values() for r,_,_ in items}
    assert not known_prompts.intersection(r["prompt"] for r in fresh),"fresh/evaluation overlap"
    suites["fresh_writing"]=[(r,legacy_fresh_score,r["family"]) for r in fresh]
    leakage=[]; changed_counts={c:0 for c in CONFIGS}
    for suite,items in suites.items():
        for r,_,family in items:
            before=routes["v3_baseline"](r["prompt"])
            for c,route in routes.items():
                after=route(r["prompt"])
                if after!=before:
                    changed_counts[c]+=1
                    if before[0]!="model" or (family!="drafting" and suite!="fresh_writing"):
                        leakage.append([c,suite,r.get("id"),family])
    if leakage: raise RuntimeError("Out-of-scope changes: "+json.dumps(leakage))
    audit=E.audit(M,[r["prompt"] for r in exact])
    if any(audit.values()): raise RuntimeError(audit)
    print("WRITING_SCOPE_PASS",json.dumps({"changed_routes":changed_counts,"gate_leakage":leakage}),flush=True)
    tok=AutoTokenizer.from_pretrained(E.BASE,revision=E.BASE_REV); tok.pad_token=tok.eos_token
    base,loading=Qwen3_5ForCausalLM.from_pretrained(E.BASE,revision=E.BASE_REV,dtype=torch.bfloat16,
        device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if loading["missing_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError("base load mismatch")
    model=PeftModel.from_pretrained(base,E.MODEL,revision=E.MODEL_REV).eval()
    model.requires_grad_(False)
    cache={}
    def generate(system,prompt):
        key=(system,prompt)
        if key not in cache:
            ids=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":prompt}],
                tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
            with torch.inference_mode():
                out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,
                    do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
            cache[key]=tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()
        return cache[key]
    results={c:{} for c in CONFIGS}; outputs={c:{} for c in CONFIGS}
    legacy_results={c:{} for c in CONFIGS}; records=[]
    for suite,items in suites.items():
        for c,route in routes.items():
            score=[0,len(items)]; families={}; legacy_score=[0,len(items)]; legacy_families={}
            review_count=0
            for i,(row,scorer,family) in enumerate(items):
                kind,value=route(row["prompt"])
                output=value if kind=="tool" else generate(value,row["prompt"])
                grading=grade_case(row,output,scorer); ok=grading["passed"]; score[0]+=ok
                legacy_ok=grading["legacy_passed"]; legacy_score[0]+=legacy_ok
                oldfam=legacy_families.setdefault(family,[0,0]); oldfam[0]+=legacy_ok; oldfam[1]+=1
                review_count+=grading["status"]=="review"
                fam=families.setdefault(family,[0,0]); fam[0]+=ok; fam[1]+=1
                outputs[c][(suite,i)]={"id":row.get("id",str(i)),"ok":ok,"output":output,"grading":grading}
                records.append({"config":c,"suite":suite,"row":{**row,"id":row.get("id",str(i))},
                    "output":output,"legacy_ok":legacy_ok,"ok":ok,"grading":grading})
            results[c][suite]={"score":score,"families":families,"review_count":review_count}
            legacy_results[c][suite]={"score":legacy_score,"families":legacy_families}
            print("WRITING_SCORE",json.dumps({"config":c,"suite":suite,"score":score}),flush=True)
    reproduced,verdict=score_comparison(results,outputs,legacy_results,audit,leakage)
    diffs={}
    for c in CONFIGS:
        if c=="v3_baseline": continue
        diffs[c]=[{"suite":k[0],"id":v["id"],"baseline_ok":outputs["v3_baseline"][k]["ok"],
            "candidate_ok":v["ok"],"baseline":outputs["v3_baseline"][k]["output"],"candidate":v["output"]}
            for k,v in outputs[c].items() if v["ok"]!=outputs["v3_baseline"][k]["ok"]]
    eligible=[c for c,v in verdict.items() if v["eligible_for_manual_review"]]
    best=max(eligible,key=lambda c:(results[c]["suite_v3"]["score"][0],results[c]["fresh_writing"]["score"][0],-sum(CONFIGS[c]))) if eligible else None
    summary={"source_commit":SOURCE_COMMIT,"source_sha256":source_sha,"model":E.MODEL,"model_revision":E.MODEL_REV,
        "grader_version":GRADER_VERSION,"scores":results,"legacy_scores":legacy_results,
        "legacy_baseline_reproduced":reproduced,"verdict":verdict,"best_candidate":best,
        "generations":len(cache),"weights_changed":False,"automatic_promotion":False,"production_ready":False,
        "development_suites":["suite_v3","probes","fresh_writing"],"new_holdout_cases":0,
        "previously_observed_writing_cases":24,
        "fresh_scores_are_automated_pending_manual_review":True}
    Path("writing-results.json").write_text(json.dumps({"summary":summary,"diffs":diffs,"records":records},indent=2))
    print("WRITING_DIFFS",json.dumps(diffs),flush=True)
    for i,row in enumerate(fresh):
        print("WRITING_FRESH_REVIEW",json.dumps({"row":row,"outputs":{c:outputs[c][("fresh_writing",i)] for c in CONFIGS}}),flush=True)
    print("WRITING_SUMMARY",json.dumps(summary),flush=True)

if __name__=="__main__": main()
