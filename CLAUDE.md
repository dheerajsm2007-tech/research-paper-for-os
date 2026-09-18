# CLAUDE.md — Non-negotiable rules

This file governs every change made to this repository, by Claude Code or anyone else.
It exists because the single most common fatal flaw in burst-time / runtime prediction
papers is feature leakage. These rules are the guardrail against that. Do not weaken,
reinterpret, or work around them without updating this file and explaining why in the
commit message.

1. **Every feature must be knowable at task arrival time.** Never use actual runtime,
   end timestamp, exit status, or peak memory as an input feature.

2. **Historical features must be computed causally**: for task *i*, only from tasks
   that COMPLETED strictly before task *i* ARRIVED.

3. **Train/test splits are always chronological, never random, never shuffled.**

4. **The prediction target is job-level runtime, not true CPU burst.** Do not rename it.

5. **Random seeds are fixed and declared in one place.**

6. **No feature may be added without a one-line justification of why it is available
   at arrival time.**
