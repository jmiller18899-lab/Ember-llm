"""Behavioral contracts for WR11's prompt-only ablation and checkpoint selection."""
import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jobs"))
import ember_writing_repair1_data as D
import ember_writing_repair10_data as OLD
import ember_writing_repair10_train as OLD_T
import ember_meaning_preservation_v3 as V3
import ember_writing_repair11_data as W
import ember_writing_repair11_train as T

def snapshot(step, passes, copies=2, force=0):
    rows = [{"id": str(i), "status": "pass" if i in passes else "fail", "reasons": []} for i in range(6)]
    return {"step": step, "copies": copies, "force_fails": force, "v3_pass": len(passes),
            "meaning": {"rows": rows}}

class WR11Tests(unittest.TestCase):
    def test_each_shortening_group_has_half_ordinary_requests(self):
        groups = {}
        for row in W.writing("train"):
            if row["family"] != "shortening": continue
            prefix = row["prompt"].split(row["source"])[0].lower()
            explicit = any(k in prefix for k in ("reminder", "force", "fact", "obligation", "remember/forget"))
            groups.setdefault(row["shortening_group"], Counter())[explicit] += 1
        self.assertEqual(len(groups), 8)
        self.assertTrue(all(counts == {False: 6, True: 6} for counts in groups.values()), groups)

    def test_only_training_prompt_and_identity_change(self):
        old = OLD.build_train_rows(D.retention("train"))
        new = W.build_train_rows(D.retention("train"))
        self.assertEqual(len(new), 416)
        self.assertEqual(len(new), T.STEPS * T.ACCUM)
        changed = 0
        for a,b in zip(old,new):
            self.assertEqual({k:v for k,v in a.items() if k not in ("id","prompt")},
                             {k:v for k,v in b.items() if k not in ("id","prompt")})
            if a["family"] != "shortening": self.assertEqual(a["prompt"], b["prompt"])
            changed += a["prompt"] != b["prompt"]
        self.assertGreater(changed, 0)
        self.assertEqual((T.LR,T.STEPS,T.ACCUM,T.SEED), (OLD_T.LR,OLD_T.STEPS,OLD_T.ACCUM,OLD_T.SEED))

    def test_fresh_development_sources_and_probe_exclusion(self):
        old = {r["source"] for r in OLD.writing("dev")}
        new = W.writing("dev")
        self.assertEqual(len(new),56)
        self.assertFalse(old & {r["source"] for r in new})
        self.assertTrue(all(r["training_allowed"] is False for r in new))
        spec=importlib.util.spec_from_file_location("wr11_probe", ROOT/"reports/probes/wr11_reminder_preflight.py")
        p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
        probe={r["source"] for r in p.cases()}
        self.assertFalse(probe & {r["source"] for r in W.writing("train")+new})

    def test_targets_shorten_and_retain_meaning(self):
        for row in W.writing("train"):
            if row["family"] != "shortening": continue
            self.assertEqual(V3.grade_case({"kind":"shortening","source":row["source"]},row["answer"],lambda r,o:True)["status"],"pass")
            for key in ("rejected","overedit"):
                self.assertNotEqual(V3.grade_case({"kind":"shortening","source":row["source"]},row[key],lambda r,o:True)["status"],"pass")

    def test_tied_quality_prefers_unchanged_baseline(self):
        base=snapshot(0,{0,1},2)
        self.assertEqual(T.choose_selected(base,[base,snapshot(13,{0,1},2)]),0)

    def test_fewer_copies_cannot_hide_lost_pass(self):
        base=snapshot(0,{0,1},3)
        self.assertEqual(T.choose_selected(base,[base,snapshot(13,{1,2},1)]),0)

    def test_aggregate_pass_loss_is_rejected(self):
        base=snapshot(0,{0,1,2},3)
        self.assertEqual(T.choose_selected(base,[base,snapshot(13,{0,1},1)]),0)

    def test_real_improvement_preserves_baseline_passes(self):
        base=snapshot(0,{0},2)
        self.assertEqual(T.choose_selected(base,[base,snapshot(13,{0,1},2),snapshot(26,{0,1,2},3)]),13)

    def test_bad_case_coverage_fails_closed(self):
        base=snapshot(0,{0},2); bad=snapshot(13,{0,1},1)
        bad["meaning"]["rows"].pop()
        with self.assertRaises(ValueError): T.choose_selected(base,[base,bad])

if __name__=="__main__": unittest.main()
