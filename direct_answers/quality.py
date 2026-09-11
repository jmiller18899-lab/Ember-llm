"""Unchanged v054 direct-quality checks; old 24 cases are development evidence."""
import re

SYSTEM = (
    "You are Ember. Answer the user's request directly and concisely. The request does not need a tool. "
    "Return only the answer to the current user turn."
)

def prompt(user: str) -> str:
    return f"<|system|>\n{SYSTEM}\n<|user|>\n{user}\n<|assistant|>\n"

def case(cid: str, family: str, user: str, check: dict) -> dict:
    return {"id": cid, "kind": "direct_response", "family": family, "user": user, "prompt": prompt(user), "check": check}

CASES = [
    # Friendly/writing.
    case("dq_greet_1", "writing", "Say hello in one friendly sentence.", {"any_terms": [["hello", "hi", "hey"]], "max_words": 20}),
    case("dq_greet_2", "writing", "Write one short sentence thanking someone for reviewing my code.", {"any_terms": [["thank", "thanks", "appreciate"], ["review", "code"]], "max_words": 24}),
    case("dq_write_3", "writing", "Write a friendly one-sentence message saying the upload succeeded.", {"any_terms": [["upload"], ["success", "succeeded", "complete", "completed"]], "max_words": 24}),
    case("dq_write_4", "writing", "Write a short tooltip for a refresh button.", {"any_terms": [["refresh", "reload", "update"]], "max_words": 18}),

    # Rewrite/title.
    case("dq_rewrite_1", "rewrite", "Rewrite as a clear title: checkout button broken on mobile.", {"all_terms": ["checkout", "mobile"], "any_terms": [["broken", "fix", "issue", "failure", "problem"]], "max_words": 12}),
    case("dq_rewrite_2", "rewrite", "Rewrite as a professional title: settings page loads really slow.", {"all_terms": ["settings"], "any_terms": [["slow", "performance", "loading", "load"]], "max_words": 12}),
    case("dq_rewrite_3", "rewrite", "Shorten this heading: user profile picture upload accessibility improvements.", {"any_terms": [["profile"], ["upload", "picture", "image"], ["accessibility", "accessible"]], "max_words": 12}),
    case("dq_rewrite_4", "rewrite", "Rewrite clearly: latest build notes need cleanup.", {"all_terms": ["build"], "any_terms": [["notes", "release"], ["cleanup", "clean", "improve"]], "max_words": 12}),

    # Explanations.
    case("dq_explain_1", "explain", "Explain what a model checkpoint is in one sentence.", {"any_terms": [["save", "saved", "snapshot"], ["model", "training", "state", "weights"]], "max_words": 36}),
    case("dq_explain_2", "explain", "Explain what a cache does in one sentence.", {"any_terms": [["store", "stores", "stored", "save", "keeps"], ["fast", "faster", "quick", "reuse", "access"]], "max_words": 36}),
    case("dq_explain_3", "explain", "Explain why software tests are useful in one sentence.", {"any_terms": [["bug", "error", "problem", "issue", "correct"], ["find", "catch", "detect", "prevent", "verify"]], "max_words": 36}),
    case("dq_explain_4", "explain", "Explain the difference between a file and a folder in one sentence.", {"all_terms": ["file", "folder"], "any_terms": [["contain", "contains", "holds", "stores"], ["data", "files", "item", "items"]], "max_words": 40}),

    # Summaries.
    case("dq_summary_1", "summarize", "Summarize in one sentence: The tests passed, the deployment completed, and no errors were reported.", {"any_terms": [["test"], ["deploy"], ["pass", "success", "error", "errors"]], "max_words": 28}),
    case("dq_summary_2", "summarize", "Summarize: The sidebar was moved to the left, its labels were shortened, and the underlying behavior did not change.", {"any_terms": [["sidebar"], ["left", "move"], ["label"], ["behavior", "function", "functionality", "unchanged"]], "max_words": 32}),
    case("dq_summary_3", "summarize", "Summarize in one sentence: The API latency dropped from 300 ms to 120 ms after the optimization.", {"any_terms": [["latency", "speed", "faster"], ["300"], ["120"], ["optim", "improve", "drop", "reduc"]], "max_words": 30}),
    case("dq_summary_4", "summarize", "Summarize: The backup finished successfully, but the notification arrived five minutes late.", {"any_terms": [["backup"], ["success", "finished", "complete"], ["notification"], ["late", "delay", "five", "5"]], "max_words": 30}),

    # Classification/sentiment: require the requested label.
    case("dq_class_1", "classify", "Label the sentiment as positive, negative, or neutral: The new interface is much easier to use.", {"label": "positive", "forbidden_labels": ["negative", "neutral"], "max_words": 12}),
    case("dq_class_2", "classify", "Label the sentiment as positive, negative, or neutral: The app crashes every time I save.", {"label": "negative", "forbidden_labels": ["positive", "neutral"], "max_words": 12}),
    case("dq_class_3", "classify", "Classify this as a question or statement: The service is online.", {"label": "statement", "forbidden_labels": ["question"], "max_words": 12}),
    case("dq_class_4", "classify", "Classify this as success, warning, or error: Deployment failed because the configuration is invalid.", {"label": "error", "forbidden_labels": ["success", "warning"], "max_words": 12}),

    # Planning/comparison.
    case("dq_plan_1", "plan_compare", "Give exactly two short steps for testing a login form.", {"min_steps": 2, "max_words": 45, "any_terms": [["login", "credential", "password", "username"], ["test", "verify", "check", "submit"]]}),
    case("dq_plan_2", "plan_compare", "Give exactly two short steps for checking whether a link works.", {"min_steps": 2, "max_words": 45, "any_terms": [["click", "open", "link"], ["verify", "check", "page", "destination"]]}),
    case("dq_compare_3", "plan_compare", "Compare JSON and CSV in two concise sentences.", {"all_terms": ["json", "csv"], "any_terms": [["structure", "structured", "key", "nested", "table", "row", "column"]], "max_words": 48}),
    case("dq_compare_4", "plan_compare", "Compare a browser tab and a browser window in two concise sentences.", {"all_terms": ["tab", "window"], "any_terms": [["browser"], ["multiple", "contain", "inside", "separate"]], "max_words": 48}),
]

LEAK_PATTERNS = [
    "what is the difference between",
    "prepare a controlled example",
    "practical mechanism used to make a system easier",
    "recent webassembly announcements",
    "the current kubernet",
]

def normalized(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.+-]+", " ", text.casefold()).split())

def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))

def generic_quality(text: str) -> dict:
    visible = ev.visible_text(text).strip()
    norm = normalized(visible)
    words = re.findall(r"\b[a-z0-9]+\b", norm)
    repeated_triplet = bool(re.search(r"\b([a-z0-9]+)(?:\s+\1){2,}\b", norm))
    unique_ratio = len(set(words)) / len(words) if words else 0.0
    leak = next((p for p in LEAK_PATTERNS if p in norm), None)
    alphabetic = sum(ch.isalpha() for ch in visible)
    printable = sum(ch.isprintable() and not ch.isspace() for ch in visible)
    alpha_ratio = alphabetic / printable if printable else 0.0
    passed = (
        2 <= len(words) <= 64
        and not repeated_triplet
        and unique_ratio >= 0.45
        and alpha_ratio >= 0.45
        and leak is None
    )
    return {
        "passed": passed,
        "word_count": len(words),
        "unique_word_ratio": unique_ratio,
        "alpha_ratio": alpha_ratio,
        "repeated_triplet": repeated_triplet,
        "leak_pattern": leak,
    }

def count_steps(text: str) -> int:
    # Prefer explicit numbered/bulleted lines; fall back to sentence count.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    explicit = sum(bool(re.match(r"^(?:\d+[.)]|[-*•])\s*", line)) for line in lines)
    if explicit:
        return explicit
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(sentences)

def semantic_check(text: str, spec: dict) -> dict:
    norm = normalized(text)
    wc = word_count(text)
    reasons = []
    max_words = int(spec.get("max_words", 64))
    if wc > max_words:
        reasons.append(f"too_long:{wc}>{max_words}")
    for term in spec.get("all_terms", []):
        if normalized(term) not in norm:
            reasons.append(f"missing:{term}")
    for alternatives in spec.get("any_terms", []):
        if not any(normalized(term) in norm for term in alternatives):
            reasons.append("missing_any:" + "/".join(alternatives))
    if "label" in spec:
        label = normalized(spec["label"])
        if label not in norm:
            reasons.append(f"missing_label:{label}")
        for bad in spec.get("forbidden_labels", []):
            if normalized(bad) in norm:
                reasons.append(f"wrong_label_present:{bad}")
    if "min_steps" in spec:
        steps = count_steps(text)
        if steps < int(spec["min_steps"]):
            reasons.append(f"too_few_steps:{steps}")
    return {"passed": not reasons, "reasons": reasons, "word_count": wc}
