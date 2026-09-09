"""Pinned, streaming source adapters for the v0.1.0 CPU corpus builder."""
from __future__ import annotations

import json

import ember_corpus_build_v010 as builder
import ember_corpus_envelope_copy_v010 as envelope_copy

CODE_LANGUAGES = {
    "Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "Go", "Rust",
    "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Shell", "PowerShell", "Lua", "Dart",
    "SQL", "HTML", "CSS", "Vue", "Svelte", "R", "Julia", "Objective-C", "Objective-C++",
    "Dockerfile", "Makefile", "CMake", "YAML", "JSON", "TOML", "Markdown",
}


def resolve_revisions(cfg: dict) -> dict:
    from huggingface_hub import HfApi
    datasets = set()
    for source in cfg["sources"].values():
        datasets.update(x for x in (source.get("dataset"), source.get("license_join_dataset")) if x)
    revisions = {}
    for dataset in sorted(datasets):
        sha = HfApi().dataset_info(dataset).sha
        if not sha:
            raise RuntimeError(f"no immutable revision for {dataset}")
        revisions[dataset] = sha
    print(json.dumps({"event": "resolved_revisions", "datasets": revisions}), flush=True)
    return revisions


def document(cfg, revisions, category, source_id, text, *, group_id=None, **provenance):
    source = cfg["sources"][category]
    return {"category": category, "source": source["dataset"] or "synthetic/envelope-copy-v0.1.0",
            "source_id": str(source_id), "group_id": str(group_id or source_id), "text": text,
            "provenance": {"license": source["license"],
                           "dataset_revision": revisions.get(source["dataset"]), **provenance}}


def smoltalk_document(row, cfg, revisions, row_no):
    """Exclude the differently formatted API subset and reject malformed rows."""
    if row.get("source") == "apigen-80k":
        return None
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    if (not any(m.get("role") == "user" for m in messages)
            or not any(m.get("role") == "assistant" for m in messages)):
        return None
    if any(m.get("role") not in ("system", "user", "assistant") or m.get("tool_calls")
           or not isinstance(m.get("content"), str) for m in messages):
        return None
    text = builder.render_openai_messages(messages)
    return document(cfg, revisions, "instruction_following", builder.digest(json.dumps(messages, sort_keys=True)),
                    text, subset=row.get("source"), source_row=row_no)


def synthetic_documents(cfg):
    generator_sha = builder.file_hash(builder.ROOT / "jobs/ember_corpus_envelope_copy_v010.py")
    for doc in envelope_copy.iter_documents(cfg):
        yield document(cfg, {}, "envelope_copy", doc["id"], doc["text"],
                       generator_sha256=generator_sha,
                       envelope_copy={k: v for k, v in doc.items() if k not in ("text", "id")})


def collect_sources(cfg, revisions, store, *, load_dataset=None):
    if load_dataset is None:
        from datasets import load_dataset
    corpus, _ = builder.legacy()
    seed = cfg["seed"]

    def rows(category, *, split="train", shuffle=True):
        source = cfg["sources"][category]
        stream = load_dataset(source["dataset"], name=source.get("config"), split=split,
                              streaming=True, revision=revisions[source["dataset"]])
        if shuffle:
            size = int(cfg.get("stream_shuffle_buffers", {}).get(category, 64))
            if size < 1:
                raise ValueError("stream shuffle buffers must be positive")
            stream = stream.shuffle(seed=seed + list(cfg["mix"]).index(category), buffer_size=size)
        return stream

    def done(*categories):
        store.db.commit()
        for category in categories:
            store.finish_category(category)
            print(json.dumps({"event": "category_complete", "category": category,
                              "candidate_chars": store.chars[category], "documents": store.counts[category]}), flush=True)

    category = "general_english"
    for index, row in enumerate(rows(category)):
        if store.full(category):
            break
        identity = str(row.get("id") or builder.digest(str(row.get("text") or "")))
        for chunk_no, chunk in enumerate(corpus.split_text_chunks(str(row.get("text") or ""),
                max_chars=cfg["max_document_chars"], min_chars=max(300, cfg["min_document_chars"]))):
            store.add(document(cfg, revisions, category, f"{identity}:{chunk_no}", corpus.ensure_eot(chunk),
                               group_id=identity, url=row.get("url"), dump=row.get("dump")))
    done(category)

    category = "code"
    for row in rows(category):
        if store.full(category):
            break
        repo_path, commit = str(row.get("repo_path") or ""), str(row.get("commit_id") or "")
        files = sorted(row.get("files") or [], key=lambda f: builder.digest(f"{repo_path}:{f.get('file_path')}:{seed}"))
        for item in files:
            if store.full(category):
                break
            licenses = [str(v) for v in item.get("detected_licenses") or [] if str(v).strip()]
            language = item.get("language")
            if (item.get("is_vendor") or item.get("license_type") != "permissive" or not licenses
                    or (language and language not in CODE_LANGUAGES)):
                continue
            group = f"{repo_path}@{commit}:{item.get('file_path')}"
            for chunk_no, chunk in enumerate(corpus.split_text_chunks(str(item.get("content") or ""),
                    max_chars=cfg["max_document_chars"], min_chars=max(120, cfg["min_document_chars"] // 2))):
                store.add(document(cfg, revisions, category, f"{group}:{chunk_no}", corpus.ensure_eot(chunk),
                                   group_id=group, repo_path=repo_path, commit_id=commit, file_path=item.get("file_path"),
                                   language=language, detected_licenses=licenses, license_type="permissive"))
    done(category)

    category = "instruction_following"
    for index, row in enumerate(rows(category)):
        if store.full(category):
            break
        doc = smoltalk_document(row, cfg, revisions, index)
        if doc and len(doc["text"]) >= cfg["min_document_chars"]:
            store.add(doc)
    done(category)

    # OASST is finite. Keep its parent lookup on disk too; never repeat rows to fill a quota.
    category = "instruction_following_human"
    store.db.execute("CREATE TABLE oasst (id TEXT PRIMARY KEY, order_key TEXT, data TEXT)")
    store.db.execute("CREATE INDEX oasst_order ON oasst(order_key)")
    for row in rows(category, shuffle=False):
        if row.get("message_id"):
            store.db.execute("INSERT OR REPLACE INTO oasst VALUES (?,?,?)", (
                row["message_id"], builder.digest(f"{row['message_id']}:{seed}"), json.dumps(row)))
    store.db.commit()
    for raw, in store.db.execute("SELECT data FROM oasst ORDER BY order_key"):
        if store.full(category):
            break
        row = json.loads(raw)
        if (row.get("role") != "assistant" or row.get("lang") != "en" or row.get("deleted")
                or row.get("review_result") is False or (row.get("rank") is not None and int(row["rank"]) > 1)):
            continue
        parent = store.db.execute("SELECT data FROM oasst WHERE id=?", (row.get("parent_id"),)).fetchone()
        parent = json.loads(parent[0]) if parent else {}
        if parent.get("role") != "prompter" or parent.get("lang") != "en" or parent.get("deleted"):
            continue
        prompt, answer = str(parent.get("text") or "").strip(), str(row.get("text") or "").strip()
        if len(prompt) < 20 or len(answer) < 40:
            continue
        text = builder.render_openai_messages([{"role": "user", "content": prompt[:7000]},
                                              {"role": "assistant", "content": answer[:9000]}])
        store.add(document(cfg, revisions, category, row["message_id"], text, group_id=row["parent_id"],
                           message_id=row["message_id"], parent_id=row["parent_id"], lang="en"))
    store.db.execute("DROP TABLE oasst")
    done(category)

    glaive = ("function_calling", "tool_result_interpretation")
    for row in rows("function_calling"):
        if all(store.full(c) for c in glaive):
            break
        messages, meta = corpus.parse_glaive_row(row)
        if len(messages) < 2:
            continue
        if meta["tool_calls"] and meta["has_tool_result_interpretation"] and not store.full(glaive[1]):
            category = glaive[1]
        elif meta["tool_calls"] and not store.full(glaive[0]):
            category = glaive[0]
        else:
            continue
        try:
            text = builder.render_openai_messages(messages, drop_tool_call_reasoning=True)
        except (ValueError, TypeError):
            store.rejected[category] += 1
            continue
        if len(text) >= cfg["min_document_chars"]:
            identity = builder.digest(str(row.get("system")) + "\n" + str(row.get("chat")))
            store.add(document(cfg, revisions, category, identity, text, **meta))
    done(*glaive)

    license_repo = cfg["sources"]["agent_trajectories"]["license_join_dataset"]
    store.db.execute("CREATE TABLE licenses (id TEXT PRIMARY KEY, name TEXT)")
    splits = load_dataset(license_repo, streaming=True, revision=revisions[license_repo])
    for split in splits.values():
        for row in split:
            if row.get("instance_id"):
                store.db.execute("INSERT OR REPLACE INTO licenses VALUES (?,?)", (row["instance_id"], row.get("license_name") or ""))
    if not store.db.execute("SELECT 1 FROM licenses LIMIT 1").fetchone():
        raise RuntimeError("SWE-rebench license lookup is empty")
    store.db.commit()
    for row in rows("agent_trajectories"):
        if store.full("agent_trajectories") and store.full("error_recovery"):
            break
        identity = str(row.get("instance_id") or "")
        license_row = store.db.execute("SELECT name FROM licenses WHERE id=?", (identity,)).fetchone()
        if not license_row or not corpus.permissive_repo_license(license_row[0]):
            continue
        recovery = corpus.openhands_has_recovery(row)
        if recovery and not store.full("error_recovery"):
            category = "error_recovery"
        elif int(row.get("resolved") or 0) == 1 and not recovery and not store.full("agent_trajectories"):
            category = "agent_trajectories"
        else:
            continue
        messages = corpus.sanitize_agent_messages(row.get("trajectory") or [], drop_tool_call_reasoning=True,
                                                  drop_intermediate_assistant_content=True)
        if len(messages) < 3:
            continue
        tools = row.get("tools") or None
        try:
            text = builder.render_agent_record({"messages": messages, "tools": tools[:30] if isinstance(tools, list) else tools})
        except (ValueError, TypeError):
            store.rejected[category] += 1
            continue
        if len(text) >= cfg["min_document_chars"]:
            store.add(document(cfg, revisions, category, str(row.get("trajectory_id") or identity), text,
                group_id=identity, instance_id=identity, repo=row.get("repo"),
                repository_license=license_row[0], resolved=int(row.get("resolved") or 0),
                exit_status=row.get("exit_status"), real_error_recovery=recovery,
                reasoning_policy="intermediate free-form reasoning omitted; structured calls/results and final answer retained"))
    store.db.execute("DROP TABLE licenses")
    done("agent_trajectories", "error_recovery")

    for doc in synthetic_documents(cfg):
        if store.full("envelope_copy"):
            break
        store.add(doc)
    done("envelope_copy")
