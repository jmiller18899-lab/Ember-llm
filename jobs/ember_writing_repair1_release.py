"""Canonical Writing Repair 1 data release. Data preparation only; no training.

Keep the base construction snapshot intact. These explicit reference corrections
are applied BEFORE generation, auditing, hashing or freezing any test set.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import ember_writing_repair1_data as _base

BASE_BUILDER_SHA256='2c3c555833c6cc782b1266c3f88ae6616f91e4dd6154290c179eb61aead5f7bd'
if hashlib.sha256(Path(_base.__file__).read_bytes()).hexdigest()!=BASE_BUILDER_SHA256:
    raise RuntimeError('Base dataset construction snapshot does not match the audited version')

_base.VERSION='ember-writing-repair1-data-v1.2'
# Every object slot in these structures takes singular agreement.
_base.POOLS['train']['objects'][-1]='viewfinder'
_base.SHORT['dev'][7]['reference']='Each of the {group} is asked to bring the {obj} to the {place} on {day}.'
_base.SHORT['dev'][7]['order']=['each','{group}','bring','{obj}']
_base.SHORT['writing_holdout'][3]['reference']='The {group} must not enter the {place} before {time}.'
_base.SHORT['writing_holdout'][3]['protect']=['{group}','{place}','before {time}']
_base.SHORT['writing_holdout'][3]['order']=['{group}','enter','{place}']
f=list(_base.RECIPIENT_FRAMES['writing_holdout'][3]); f[1]=f[1].replace('about a {obj}', 'about the {obj}')
_base.RECIPIENT_FRAMES['writing_holdout'][3]=tuple(f)

# Export the tested construction and audit API using the corrected recipe.
from ember_writing_repair1_data import *

def write_package(root, splits):
    manifest=_base.write_package(root,splits)
    manifest['canonical_builder']='jobs/ember_writing_repair1_release.py'
    manifest['base_builder_sha256']=BASE_BUILDER_SHA256
    manifest['reference_corrections']='Plural-group grammar, article and object agreement, before test-set freeze'
    (Path(root)/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('writing-repair1-package'))
    args=parser.parse_args()
    print(json.dumps(write_package(args.output,build_splits()),indent=2))
