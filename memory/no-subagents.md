---
name: no-subagents
description: Work directly — do not delegate to subagents/workflow fan-outs on this project
metadata:
  type: feedback
---

Do the work in the main session; don't spawn subagents or Workflow fan-outs (told 2026-08-30, right after a 14-agent literature-sweep workflow).

**Why:** stated as a standing preference, no reason given — the user tracks the work through the live logs and the direct tool calls, and delegated agents put that behind a curtain.

**How to apply:** research, searching, and multi-step analysis all happen inline — my own WebSearch/WebFetch, Bash, and file tools. Long jobs still run server-side under nohup with pollers (that's not delegation, that's [[pa-limb-pipeline]]'s normal workflow). If a task looks big enough to want parallel agents, just do it sequentially and say so.
