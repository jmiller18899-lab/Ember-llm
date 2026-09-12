"""Emit exact JSON evidence in bounded log chunks as well as Actions artifacts."""
import base64
import json
from pathlib import Path
import zlib
from .runtime import sha256


def emit_json(path, name):
    path = Path(path)
    raw = path.read_bytes()
    json.loads(raw)
    encoded = base64.b64encode(zlib.compress(raw)).decode()
    chunks = [encoded[i:i+4000] for i in range(0, len(encoded), 4000)]
    digest = sha256(path)
    for index, chunk in enumerate(chunks):
        print(json.dumps({"event": "ember_evidence_chunk", "name": name, "sha256": digest,
                          "index": index, "count": len(chunks), "data": chunk}), flush=True)
