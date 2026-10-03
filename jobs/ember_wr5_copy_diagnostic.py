# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0","transformers==5.17.0","peft==0.20.0","accelerate==1.15.0","huggingface-hub==1.31.0","jinja2==3.1.6"]
# ///
"""Read-only WR5 checkpoint replay: no optimizer, training, or promotion."""
import argparse, hashlib, importlib.util, json, os, re, tempfile
from collections import Counter
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "Jmiller18899/ember-qwen3.5-4b-writing-repair5-20261003"
REV = "4a72d7d5a2811d427329cf650f44ec469d39ca9f"
OUT = "Jmiller18899/ember-wr5-copying-diagnostic-20261003"
TRAIN_COMMIT = "4e406a094c0494e684eea9850c70242c441bc3ef"
BRANCH = "experiment/ember-wr5-copying-investigation-20261003"
CHECKPOINTS = [(0,"checkpoint-0"),(32,"checkpoints/step-32"),(64,"checkpoints/step-64"),(96,"checkpoints/step-96"),(128,"checkpoints/step-128")]
TARGET_IDS = {"promo_v2/draft-short-00","promo_v2/draft-short-05","fresh_writing/fresh-shorten-03","suite_v3/drafting-07"}
GOLD = {
 "promo_v2/draft-short-00": "Please ensure everyone gathers at the loading dock before 7.",
 "promo_v2/draft-short-05": "Please ensure everyone gathers at the loading dock before 8.",
 "fresh_writing/fresh-shorten-03": "Please remember the spare key must remain in locker 27 until Monday.",
 "suite_v3/drafting-07": "Hi Priya, I picked up your notebook and can bring it by Wednesday."
}

def setup():
    api=HfApi(token=os.environ["HF_TOKEN"])
    if api.whoami().get("name")!="Jmiller18899": raise ValueError("Wrong account")
    work=Path(tempfile.mkdtemp(prefix="ember-copy-diag-"))
    def raw(name):
        return Path(hf_hub_download(REPO,name,revision=REV)).read_bytes()
    def saved(name): return json.loads(raw(name))
    launch=saved("launch.json")
    for name,expected in [("ember_writing_repair5_train.py",launch["trainer_sha256"]),("ember_writing_repair5_data.py",launch["data_code_sha256"])]:
        content=raw("source/"+name)
        if hashlib.sha256(content).hexdigest()!=expected: raise ValueError("Source hash mismatch")
        (work/name).write_bytes(content)
    spec=importlib.util.spec_from_file_location("wr5_original",work/"ember_writing_repair5_train.py")
    R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
    os.environ["WR5_CODE_COMMIT"]=TRAIN_COMMIT
    prepared=R.prepare()
    folder,T,W,G,E,M,suites,rows,dev,tok,route,encoded,run_spec=prepared
    for k,v in run_spec.items():
        if launch.get(k)!=v: raise ValueError("Pinned run specification mismatch: "+k)
    actual_training=raw("train/sft_train_only.jsonl")
    if actual_training!=W.sft_bytes(rows): raise ValueError("Saved training export differs")
    before=saved("evidence/baseline-744.json")["records"]
    after=saved("evidence/candidate-744.json")["records"]
    db=saved("evidence/dev-before.json");da=saved("evidence/dev-after.json")
    completion=saved("evidence/final-report.json")
    lookup={(s,row.get("id",str(i))):(row,scorer) for s,items in suites.items() for i,(row,scorer,fam) in enumerate(items)}
    cases=[]
    for b,a in zip(before,after):
        key=b["suite"]+"/"+b["id"];src=W.shortening_source(b["row"])
        if src or key=="suite_v3/drafting-07":
            row,scorer=lookup[(b["suite"],b["id"])]
            if row!=b["row"] or row!=a["row"]:raise ValueError("Benchmark input differs")
            cases.append(dict(id=key,group="benchmark_shortening" if src else "recipient_regression",
                prompt=row["prompt"],source=src,row=row,scorer=scorer,before=b["output"],after=a["output"]))
    for b,a in zip(db,da):
        if b["family"]=="shortening" or b["id"] in ("wr5-dev-recipient-01","wr5-dev-recipient-05"):
            cases.append(dict(id=b["id"],group="development_shortening" if b["family"]=="shortening" else "recipient_control",
                prompt=b["prompt"],source=b["source"] if b["family"]=="shortening" else None,
                before=b["output"],after=a["output"]))
    if len(cases)!=65 or len({c["id"] for c in cases})!=65:raise ValueError("Diagnostic coverage changed")
    for c in cases:
        kind,system=route(c["prompt"])
        if kind!="model":raise ValueError("Diagnostic case bypasses model")
        c["system"]=system
    label_counts=Counter()
    for row,item in zip(rows,encoded):
        kind,system=route(row["prompt"])
        system=system if kind=="model" else M.O.SYSTEM
        prefix=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":row["prompt"]}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        answer=tok.encode(row["answer"],add_special_tokens=False)+[tok.convert_tokens_to_ids("<|im_end|>")]
        if item["input_ids"]!=prefix+answer or item["labels"]!=[-100]*len(prefix)+answer:
            raise ValueError("Loss target/prompt masking mismatch")
        label_counts[row["family"]]+=len(answer)
    short=[r for r in rows if r["family"]=="shortening"]
    pronoun=[r for r in rows if r.get("relation")=="recipient" and r.get("representation")=="pronoun"]
    coverage=Counter((r["owner_pronoun"],(re.search(r"\b(can|should|might|will)\b",r["source"]) or re.match("none","none")).group()) for r in pronoun)
    audit=dict(training_rows=len(rows),family_examples=dict(Counter(r["family"] for r in rows)),
        supervised_tokens_by_family=dict(label_counts),answer_only_label_rows_verified=len(encoded),
        shortening_rows=len(short),identical_shortening_targets=sum(W.norm(r["source"])==W.norm(r["answer"]) for r in short),
        shortening_reference_mean_ratio=sum(len(r["answer"])/len(r["source"]) for r in short)/len(short),
        shortening_groups=dict(Counter(r["shortening_group"] for r in short)),
        short_training_templates=len({r["structure"] for r in short}),
        little_to_cut_templates=len({r["structure"] for r in short if r["shortening_group"]=="little"}),
        little_to_cut_common_suffix=all(r["source"].endswith(", if you please.") for r in short if r["shortening_group"]=="little"),
        recipient_pronoun_modal_coverage={"/".join(k):v for k,v in coverage.items()},
        original_input_revision=REV,original_training_commit=TRAIN_COMMIT,case_count=len(cases),
        checkpoints=[s for s,_ in CHECKPOINTS],new_training=False,final_holdouts_loaded=False)
    return api,R,prepared,cases,completion,audit,work

def launch_job(audit):
    if os.environ.get("GITHUB_REF")!="refs/heads/"+BRANCH or os.environ.get("GITHUB_RUN_ATTEMPT")!="1":
        raise ValueError("Only the explicit diagnostic branch's first attempt may submit")
    api=HfApi(token=os.environ["HF_TOKEN"])
    if api.repo_exists(OUT):raise ValueError("Diagnostic output exists; duplicate refused")
    api.create_repo(OUT,private=True)
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    reservation=dict(audit,diagnostic_code_commit=os.environ["COPY_CODE_COMMIT"],diagnostic_script_sha256=script_hash)
    api.upload_file(repo_id=OUT,path_in_repo="reservation.json",path_or_fileobj=json.dumps(reservation,indent=2).encode())
    job=api.run_uv_job(script=str(Path(__file__).resolve()),script_args=["--infer"],python="3.11",flavor="l4x1",timeout="45m",
        env={"COPY_CODE_COMMIT":os.environ["COPY_CODE_COMMIT"],"TOKENIZERS_PARALLELISM":"false"},
        secrets={"HF_TOKEN":os.environ["HF_TOKEN"]},labels={"name":"ember-wr5-copying-diagnostic-20261003"})
    receipt=dict(job_id=job.id,url=job.url,output_repo=OUT,inference_only=True,automatic_retry=False)
    api.upload_file(repo_id=OUT,path_in_repo="launch-submission.json",path_or_fileobj=json.dumps(receipt,indent=2).encode())
    print("COPY_GPU_SUBMITTED",json.dumps(receipt),flush=True)
    with open(os.environ["GITHUB_OUTPUT"],"a") as f:f.write("job_id="+job.id+"\n")

def infer(api,R,prepared,cases,completion,audit,work):
    import torch
    from peft import PeftModel
    from transformers import Qwen3_5ForCausalLM,set_seed
    if not torch.cuda.is_available():raise ValueError("GPU required for replay")
    if not api.model_info(OUT).private:raise ValueError("Diagnostic output must stay private")
    reservation=json.loads(Path(hf_hub_download(OUT,"reservation.json")).read_text())
    if reservation["diagnostic_script_sha256"]!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Diagnostic script changed")
    if api.file_exists(OUT,"final-report.json"):raise ValueError("Diagnostic already completed")
    folder,T,W,G,E,M,suites,rows,dev,tok,route,encoded,run_spec=prepared
    set_seed(431)
    base,loading=Qwen3_5ForCausalLM.from_pretrained(T.BASE,revision=T.BASE_REV,dtype=torch.bfloat16,
        device_map={"":0},output_loading_info=True,key_mapping={r"^model.language_model\.":"model."})
    if loading["missing_keys"] or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError("Base weights did not load exactly")
    model=PeftModel.from_pretrained(base,REPO,revision=REV,subfolder="checkpoint-0",is_trainable=False)
    def upload(name,value):
        api.upload_file(repo_id=OUT,path_in_repo=name,path_or_fileobj=json.dumps(value,indent=2,ensure_ascii=False).encode(),
            commit_message="Persist read-only copying diagnostic")
    def prefix(system,prompt):
        return tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":prompt}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors="pt",return_dict=False).to(model.device)
    def generate(system,prompt):
        ids=prefix(system,prompt)
        with torch.inference_mode():
            out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,
                use_cache=True,pad_token_id=tok.pad_token_id)
        return tok.decode(out[0,ids.shape[-1]:],skip_special_tokens=True).strip()
    def nll(system,prompt,answer):
        ids=prefix(system,prompt)[0].tolist()
        encoded=R.encode_ids(ids,tok.encode(answer,add_special_tokens=False),tok.convert_tokens_to_ids("<|im_end|>"),384)
        x=torch.tensor([encoded["input_ids"]],device=model.device)
        y=torch.tensor([encoded["labels"]],device=model.device)
        count=sum(t!=-100 for t in encoded["labels"])
        with torch.inference_mode():
            loss=model(input_ids=x,attention_mask=torch.ones_like(x),labels=y,use_cache=False).loss.item()
        return dict(mean_nll=loss,total_nll=loss*count,target_tokens=count)
    def digest(adapter):
        h=hashlib.sha256();marker="."+adapter+"."
        for name,param in model.named_parameters():
            if "lora_" in name and marker in name:
                h.update(name.replace(marker,".default.").encode())
                h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    upload("data-audit.json",audit)
    summaries=[];variants=[];likelihoods=[];endpoint_mismatches=[]
    for step,subfolder in CHECKPOINTS:
        adapter="default" if step==0 else "step_"+str(step)
        if step:
            model.load_adapter(REPO,adapter_name=adapter,revision=REV,subfolder=subfolder,is_trainable=False)
            model.set_adapter(adapter)
        model.requires_grad_(False);model.eval();model.config.use_cache=True
        current_digest=digest(adapter)
        expected_digest=completion["starting_digest"] if step==0 else completion["candidate_digest"] if step==128 else None
        if expected_digest and current_digest!=expected_digest:raise ValueError("Endpoint adapter digest mismatch")
        records=[]
        for case in cases:
            output=generate(case["system"],case["prompt"])
            record=dict(case_id=case["id"],group=case["group"],prompt=case["prompt"],source=case["source"],output=output,
                verbatim_copy=W.norm(output)==W.norm(case["source"]) if case["source"] else False,
                not_shorter=len(output)>=len(case["source"]) if case["source"] else False)
            if "row" in case:record["grading"]=G.grade_case(case["row"],output,case["scorer"])
            if step in (0,128):
                expected=case["before"] if step==0 else case["after"]
                if output!=expected:endpoint_mismatches.append(dict(step=step,id=case["id"],expected=expected,actual=output))
            records.append(record)
        summary=dict(step=step,adapter_digest=current_digest,groups={})
        for group in ("benchmark_shortening","development_shortening","recipient_regression","recipient_control"):
            selected=[r for r in records if r["group"]==group]
            summary["groups"][group]=dict(rows=len(selected),verbatim_copies=sum(r["verbatim_copy"] for r in selected),
                not_shorter=sum(r["not_shorter"] for r in selected),
                passed=sum(r.get("grading",{}).get("passed",False) for r in selected))
        for case in cases:
            if case["id"] in TARGET_IDS:
                bad=case["source"] if case["source"] else case["after"]
                good=nll(case["system"],case["prompt"],GOLD[case["id"]]);copied=nll(case["system"],case["prompt"],bad)
                likelihoods.append(dict(step=step,id=case["id"],valid_answer=GOLD[case["id"]],bad_answer=bad,
                    valid=good,bad=copied,valid_minus_bad_logprob=copied["total_nll"]-good["total_nll"]))
            if step in (0,128) and case["id"] in TARGET_IDS and case["source"]:
                for i,wrapper in enumerate(W.WRAPPERS):
                    prompt=wrapper.format(s=case["source"])
                    kind,system=route(prompt)
                    if kind!="model":raise ValueError("Prompt variant bypasses model")
                    output=generate(system,prompt)
                    variants.append(dict(step=step,id=case["id"],wrapper=i,prompt=prompt,output=output,
                        verbatim_copy=W.norm(output)==W.norm(case["source"]),not_shorter=len(output)>=len(case["source"])))
        upload("checkpoint-"+str(step)+".json",dict(summary=summary,records=records))
        upload("progress.json",dict(completed_checkpoint=step,planned_checkpoints=[s for s,_ in CHECKPOINTS]))
        summaries.append(summary)
        print("COPY_CHECKPOINT_COMPLETE",json.dumps(summary),flush=True)
    final=dict(data_audit=audit,checkpoint_summaries=summaries,prompt_variants=variants,teacher_forced_likelihoods=likelihoods,
        endpoint_mismatches=endpoint_mismatches,endpoints_reproduced=not endpoint_mismatches,
        inference_only=True,final_holdouts_loaded=False,automatic_promotion=False,
        likelihood_note="Diagnostic comparison strings only; total log probabilities are length-sensitive and are not a semantic score.")
    upload("final-report.json",final)
    print("COPY_DIAGNOSTIC_COMPLETE",json.dumps(dict(endpoints_reproduced=not endpoint_mismatches,checkpoints=len(summaries))),flush=True)
    if endpoint_mismatches:raise ValueError("Saved endpoint generations did not reproduce; causal interpretation blocked")

def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight",action="store_true");group.add_argument("--launch",action="store_true");group.add_argument("--infer",action="store_true")
    args=parser.parse_args()
    api,R,prepared,cases,completion,audit,work=setup()
    print("COPY_PREFLIGHT_PASS",json.dumps(audit),flush=True)
    if args.launch:launch_job(audit)
    elif args.infer:infer(api,R,prepared,cases,completion,audit,work)

if __name__=="__main__":main()
