"""Memory subsystem for mai-gram.

Active modules:
- messages: raw message storage and retrieval
- knowledge_base: wiki-style fact storage
- manager: high-level orchestrator

Quarantined (see ``consolidation/`` subpackage):
- summarizer: daily/weekly/monthly summarization
- forgetting: summary consolidation engine
- summaries: summary file storage

The consolidation subsystem is architecturally complete and tested but not
invoked from production paths.  It is isolated in its own subpackage so that
the active memory core remains small and coverage-enforced.
"""
