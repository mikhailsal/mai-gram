"""Memory consolidation subsystem (quarantined -- not invoked from production paths).

This package contains the daily/weekly/monthly summarization pipeline and the
forgetting engine that prunes lower-level summaries once higher-level ones exist.
The modules are well-tested and architecturally sound, but the production runtime
does not currently invoke them.

They are isolated here so that:
1. They do not contribute cognitive overhead to the active memory subsystem.
2. They are clearly separated from the coverage-enforced production core.
3. Contributors can tell at a glance that these modules are preserved capability,
   not actively supported behavior.

To activate this subsystem, wire ``MemorySummarizer`` and ``ForgettingEngine``
into the ``MemoryManager`` or a dedicated consolidation scheduler, add the package
to the coverage-enforced set, and remove the quarantine boundary.
"""

from mai_gram.memory.consolidation.forgetting import ForgettingEngine
from mai_gram.memory.consolidation.summaries import SummaryStore
from mai_gram.memory.consolidation.summarizer import MemorySummarizer

__all__ = ["ForgettingEngine", "MemorySummarizer", "SummaryStore"]
