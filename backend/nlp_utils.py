"""Rule-based NLP helpers: entity extraction (regex) and the remediation knowledge base."""
import re

PATTERNS = {
    "timestamps": r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}",
    "http_status": r"\b(?:[45]\d\d)\b",
    "latency": r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|seconds)\b",
    "percentages": r"\b\d+(?:\.\d+)?\s?%",
    "memory_sizes": r"\b\d+(?:\.\d+)?\s?(?:GiB|GB|MB|Gi|Mi)\b",
    "exceptions": r"\b[A-Z][A-Za-z]*(?:Error|Exception|Timeout)\b",
    "request_ids": r"\breq-\d+\b",
    "services": r"\[([a-z\-]+)\]",
}


def extract_entities(text: str) -> dict:
    out = {}
    for name, pat in PATTERNS.items():
        found = re.findall(pat, text)
        # keep order, drop duplicates
        seen = list(dict.fromkeys(found))
        if seen:
            out[name] = seen
    return out


KB = {
    "hallucination": {
        "title": "Hallucination / ungrounded output",
        "severity": "medium",
        "root_causes": ["Model answers without enough grounding context", "Prompt allows free-form guessing", "No post-generation fact verification"],
        "actions": ["Add retrieval or a trusted source and require citations", "Lower temperature and add 'say I don't know' instructions", "Add a groundedness / citation validator before returning answers"],
        "prevention": "Track a groundedness metric per release and alert when it drops.",
    },
    "prompt_injection": {
        "title": "Prompt injection / jailbreak attempt",
        "severity": "high",
        "root_causes": ["Untrusted text treated as instructions", "System prompt not isolated from user or retrieved content"],
        "actions": ["Quarantine the request and review the source (user input or retrieved document)", "Separate instructions from data; sanitize retrieved text", "Add an input/output safety classifier and least-privilege tool access"],
        "prevention": "Red-team regularly with known injection patterns.",
    },
    "rag_retrieval_failure": {
        "title": "RAG retrieval failure",
        "severity": "medium",
        "root_causes": ["Stale or mismatched embeddings / index", "Poor chunking", "Missing documents in the knowledge base"],
        "actions": ["Verify the embedding model and vector dimension match the index", "Re-index and re-chunk with overlap", "Add a fallback answer when retrieval returns no relevant context"],
        "prevention": "Monitor recall@k and empty-retrieval rate.",
    },
    "timeout_latency": {
        "title": "Timeout / latency degradation",
        "severity": "high",
        "root_causes": ["Overloaded inference server or deep queue", "Slow downstream dependency", "Timeouts set too tight for model generation"],
        "actions": ["Check queue depth and autoscale or shed load", "Enable streaming and set sensible per-stage timeouts", "Use a smaller or quantized model for latency-critical paths"],
        "prevention": "Alert on p95 latency and queue depth before users notice.",
    },
    "rate_limit_quota": {
        "title": "Rate limit / quota exhaustion",
        "severity": "medium",
        "root_causes": ["Burst traffic above provider limits", "Free-tier or billing quota used up"],
        "actions": ["Add exponential backoff with jitter and honour retry-after", "Queue and batch requests; cache repeated prompts", "Review quota and consider a local open-source model fallback"],
        "prevention": "Track tokens per minute against the limit.",
    },
    "resource_exhaustion": {
        "title": "Resource exhaustion (memory / GPU / disk)",
        "severity": "critical",
        "root_causes": ["Batch size or model too large for available memory", "Memory leak in a long-running worker", "Disk filled by checkpoints or logs"],
        "actions": ["Reduce batch size or use a smaller / quantized model", "Set container limits and restart policy; profile for leaks", "Clean up old checkpoints and rotate logs"],
        "prevention": "Dashboards and alerts on memory, VRAM and disk usage.",
    },
    "data_drift": {
        "title": "Data / concept drift",
        "severity": "medium",
        "root_causes": ["Production inputs differ from training distribution", "Upstream schema or unit change"],
        "actions": ["Compare live vs training feature distributions (PSI / KS test)", "Validate upstream schema and units", "Schedule retraining on recent data with human review"],
        "prevention": "Continuous drift monitoring with thresholds per feature.",
    },
    "tool_call_error": {
        "title": "Agent tool-call failure",
        "severity": "high",
        "root_causes": ["Malformed or schema-invalid tool arguments", "No error handling / retry cap in the agent loop", "Tool or external API unavailable"],
        "actions": ["Validate tool arguments against a strict schema", "Cap max iterations and detect repeated identical calls", "Return structured errors so the agent can recover or escalate"],
        "prevention": "Log every tool call with inputs, outputs and latency.",
    },
    "config_auth_error": {
        "title": "Configuration / authentication error",
        "severity": "high",
        "root_causes": ["Expired or wrong credentials", "Missing environment variables", "Incorrect endpoint, region, or permissions"],
        "actions": ["Rotate and re-check the API key / token", "Verify environment variables and config files at startup", "Review IAM roles and endpoint settings"],
        "prevention": "Add a startup config self-check and secret expiry alerts.",
    },
    "unknown": {
        "title": "Unclear / unknown failure",
        "severity": "unknown",
        "root_causes": [
            "The log does not resemble the failure types this model was trained on",
            "The wording may be unusual, very short, or not an error at all",
        ],
        "actions": [
            "Read the log yourself and check the closest guess shown above before trusting it",
            "If you know the real cause, add this line to datasets/custom_logs.csv with the right label and retrain",
        ],
        "prevention": "Keep growing your own labelled log examples so the model learns your system's wording.",
    },
}
