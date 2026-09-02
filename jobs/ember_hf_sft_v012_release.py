# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=1.4",
#   "sentencepiece>=0.2",
#   "torch>=2.4",
# ]
# ///
"""Release wrapper: final v0.0.12 trainer plus corrected inherited metadata."""
from __future__ import annotations
import urllib.request

FINAL_WRAPPER_URL = "https://raw.githubusercontent.com/jmiller18899-lab/Ember-llm/a012420902b7c77a6d08dfa329aa91f2adceffba/jobs/ember_hf_sft_v012_final.py"
outer = urllib.request.urlopen(FINAL_WRAPPER_URL, timeout=30).read().decode("utf-8")
old_exec = 'exec(compile(source, WRAPPER_URL, "exec"), {"__name__": "__main__", "__file__": WRAPPER_URL})'
new_exec = '''needle = 'source = source.replace("v0.0.11", "v0.0.12")'\nreplacement = 'source = source.replace("v0.0.11", "v0.0.12")\\nsource = source.replace(\\'"0.0.11"\\', \\'"0.0.12"\\')'\nif needle not in source:\n    raise RuntimeError("v0.0.12 version transform hook changed unexpectedly")\nsource = source.replace(needle, replacement, 1)\nexec(compile(source, WRAPPER_URL, "exec"), {"__name__": "__main__", "__file__": WRAPPER_URL})'''
if old_exec not in outer:
    raise RuntimeError("final v0.0.12 wrapper exec hook changed unexpectedly")
outer = outer.replace(old_exec, new_exec, 1)
exec(compile(outer, FINAL_WRAPPER_URL, "exec"), {"__name__": "__main__", "__file__": FINAL_WRAPPER_URL})
