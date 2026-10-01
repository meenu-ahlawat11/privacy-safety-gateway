# Project: Privacy & Safety Gateway for Generative AI

A FastAPI proxy that sits between a user and an LLM API. It detects
sensitive data and prompt injection, applies policy (mask/redact/
tokenize/block), logs to an audit trail, then forwards the sanitized prompt.

## Rules
- Python 3.11+, type hints everywhere, FastAPI + Pydantic
- Every detector returns Detection(start, end, type, confidence, source)
- NEVER log or store raw sensitive values; store types, counts, hashes only
- Use only synthetic fake data in tests
- Every module needs pytest tests, including false-positive cases
- Implement the trie/Aho-Corasick and interval merging by hand (DSA project)
- Build one module at a time; do not implement anything not asked for

## Structure
See folder layout in the repo. Pipeline order:
detect -> merge overlaps -> policy -> protect -> audit -> LLM -> restore