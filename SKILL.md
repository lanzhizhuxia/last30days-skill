---
name: last30daystest
description: TEST VERSION - Research a topic from the last 30 days on Reddit + X + Web with Bird CLI support.
argument-hint: "[topic] for [tool]" or "[topic]"
context: fork
agent: Explore
disable-model-invocation: true
allowed-tools: Bash, Read, Write, AskUserQuestion, WebSearch
---

# last30daystest: TEST VERSION with Bird CLI Support

This is the TEST version of /last30days with Bird CLI integration for free X/Twitter search.

**CLEAN MODE:** This test skill ignores your existing ~/.config/last30days/.env so you can experience the Bird CLI installation flow from scratch.

**What's new:**
- Bird CLI support for free X/Twitter search (no API key needed)
- Uses browser cookies for authentication
- Interactive prompt to install Bird if not found
- Falls back to WebSearch if Bird not available (xAI keys ignored in test mode)

Research ANY topic across Reddit, X, and the web. Surface what people are actually discussing, recommending, and debating right now.

Use cases:
- **Prompting**: "photorealistic people in Nano Banana Pro", "Midjourney prompts", "ChatGPT image generation" → learn techniques, get copy-paste prompts
- **Recommendations**: "best Claude Code skills", "top AI tools" → get a LIST of specific things people mention
- **News**: "what's happening with OpenAI", "latest AI announcements" → current events and updates
- **General**: any topic you're curious about → understand what the community is saying

## CRITICAL: Parse User Intent

Before doing anything, parse the user's input for:

1. **TOPIC**: What they want to learn about (e.g., "web app mockups", "Claude Code skills", "image generation")
2. **TARGET TOOL** (if specified): Where they'll use the prompts (e.g., "Nano Banana Pro", "ChatGPT", "Midjourney")
3. **QUERY TYPE**: What kind of research they want:
   - **PROMPTING** - "X prompts", "prompting for X", "X best practices" → User wants to learn techniques and get copy-paste prompts
   - **RECOMMENDATIONS** - "best X", "top X", "what X should I use", "recommended X" → User wants a LIST of specific things
   - **NEWS** - "what's happening with X", "X news", "latest on X" → User wants current events/updates
   - **GENERAL** - anything else → User wants broad understanding of the topic

Common patterns:
- `[topic] for [tool]` → "web mockups for Nano Banana Pro" → TOOL IS SPECIFIED
- `[topic] prompts for [tool]` → "UI design prompts for Midjourney" → TOOL IS SPECIFIED
- Just `[topic]` → "iOS design mockups" → TOOL NOT SPECIFIED, that's OK
- "best [topic]" or "top [topic]" → QUERY_TYPE = RECOMMENDATIONS
- "what are the best [topic]" → QUERY_TYPE = RECOMMENDATIONS

**IMPORTANT: Do NOT ask about target tool before research.**
- If tool is specified in the query, use it
- If tool is NOT specified, run research first, then ask AFTER showing results

**Store these variables:**
- `TOPIC = [extracted topic]`
- `TARGET_TOOL = [extracted tool, or "unknown" if not specified]`
- `QUERY_TYPE = [RECOMMENDATIONS | NEWS | HOW-TO | GENERAL]`

---

## Setup Check

The skill works in multiple modes based on available sources:

1. **Bird Mode** (free): X via Bird CLI + Reddit via OpenAI + WebSearch
2. **Full Mode** (both API keys): Reddit + X via xAI + WebSearch
3. **Partial Mode** (one key): Reddit-only or X-only + WebSearch
4. **Web-Only Mode** (no keys): WebSearch only

**Bird CLI is the preferred X source** - free, uses your browser session.

### First-Time Setup

**Option 1: Install Bird CLI (Recommended - Free X search)**

The script will prompt to install Bird if not found. Or install manually:
```bash
npm install -g @steipete/bird
```

Then log into X (twitter.com) in your browser. Bird uses your browser session.

**Option 2: API Keys (Optional)**

If you want API-based access:
```bash
mkdir -p ~/.config/last30days
cat > ~/.config/last30days/.env << 'ENVEOF'
# last30days API Configuration
# All keys are optional - Bird CLI or WebSearch fallback available

# For Reddit research (uses OpenAI's web_search tool)
OPENAI_API_KEY=

# For X/Twitter research (uses xAI's x_search tool - fallback if no Bird)
XAI_API_KEY=
ENVEOF

chmod 600 ~/.config/last30days/.env
```

**DO NOT stop if no keys are configured.** Proceed with available sources.

---

## Research Execution

**IMPORTANT: The script handles source detection automatically.** Run it and check the output.

**Step 1: Run the research script**
```bash
LAST30DAYS_CONFIG_DIR="" python3 ~/.claude/skills/last30daystest/scripts/last30days.py "$ARGUMENTS" --emit=compact 2>&1
```

The script will automatically:
- Check for Bird CLI (free X search)
- Offer to install Bird if not found and npm available
- Detect API keys
- Run Reddit/X searches with best available source
- Signal if WebSearch is needed

**Step 2: Check the output mode**

The script output will indicate the mode:
- **"Mode: both"** - Has both Reddit and X sources
- **"Mode: reddit-only"** or **"Mode: x-only"** - Has one source
- **"Mode: web-only"** - No API keys or Bird, Claude must do ALL research via WebSearch

**Step 3: Do WebSearch**

For **ALL modes**, do WebSearch to supplement (or provide all data in web-only mode).

Choose search queries based on QUERY_TYPE:

**If RECOMMENDATIONS** ("best X", "top X", "what X should I use"):
- Search for: `best {TOPIC} recommendations`
- Search for: `{TOPIC} list examples`
- Search for: `most popular {TOPIC}`
- Goal: Find SPECIFIC NAMES of things, not generic advice

**If NEWS** ("what's happening with X", "X news"):
- Search for: `{TOPIC} news 2026`
- Search for: `{TOPIC} announcement update`
- Goal: Find current events and recent developments

**If PROMPTING** ("X prompts", "prompting for X"):
- Search for: `{TOPIC} prompts examples 2026`
- Search for: `{TOPIC} techniques tips`
- Goal: Find prompting techniques and examples to create copy-paste prompts

**If GENERAL** (default):
- Search for: `{TOPIC} 2026`
- Search for: `{TOPIC} discussion`
- Goal: Find what people are actually saying

For ALL query types:
- **USE THE USER'S EXACT TERMINOLOGY** - don't substitute or add tech names based on your knowledge
- EXCLUDE reddit.com, x.com, twitter.com (covered by script)
- INCLUDE: blogs, tutorials, docs, news, GitHub repos
- **DO NOT output "Sources:" list** - this is noise, we'll show stats at the end

**Depth options** (passed through from user's command):
- `--quick` → Faster, fewer sources (8-12 each)
- (default) → Balanced (20-30 each)
- `--deep` → Comprehensive (50-70 Reddit, 40-60 X)

---

## Judge Agent: Synthesize All Sources

**After all searches complete, internally synthesize (don't display stats yet):**

The Judge Agent must:
1. Weight Reddit/X sources HIGHER (they have engagement signals: upvotes, likes)
2. Weight WebSearch sources LOWER (no engagement data)
3. Identify patterns that appear across ALL three sources (strongest signals)
4. Note any contradictions between sources
5. Extract the top 3-5 actionable insights

**Do NOT display stats here - they come at the end, right before the invitation.**

---

## FIRST: Internalize the Research

**CRITICAL: Ground your synthesis in the ACTUAL research content, not your pre-existing knowledge.**

Read the research output carefully. Pay attention to:
- **Exact product/tool names** mentioned
- **Specific quotes and insights** from the sources - use THESE, not generic knowledge
- **What the sources actually say**, not what you assume the topic is about

### If QUERY_TYPE = RECOMMENDATIONS

**CRITICAL: Extract SPECIFIC NAMES, not generic patterns.**

When user asks "best X" or "top X", they want a LIST of specific things:
- Scan research for specific product names, tool names, project names, skill names, etc.
- Count how many times each is mentioned
- Note which sources recommend each (Reddit thread, X post, blog)
- List them by popularity/mention count

### For all QUERY_TYPEs

Identify from the ACTUAL RESEARCH OUTPUT:
- **PROMPT FORMAT** - Does research recommend JSON, structured params, natural language, keywords?
- The top 3-5 patterns/techniques that appeared across multiple sources
- Specific keywords, structures, or approaches mentioned BY THE SOURCES
- Common pitfalls mentioned BY THE SOURCES

---

## THEN: Show Summary + Invite Vision

**Display in this EXACT sequence:**

**FIRST - What I learned (based on QUERY_TYPE):**

**If RECOMMENDATIONS** - Show specific things mentioned:
```
Most mentioned:
1. [Specific name] - mentioned {n}x (r/sub, @handle, blog.com)
2. [Specific name] - mentioned {n}x (sources)
3. [Specific name] - mentioned {n}x (sources)

Notable mentions: [other specific things with 1-2 mentions]
```

**If PROMPTING/NEWS/GENERAL** - Show synthesis and patterns:
```
What I learned:

[2-4 sentences synthesizing key insights FROM THE ACTUAL RESEARCH OUTPUT.]

KEY PATTERNS I'll use:
1. [Pattern from research]
2. [Pattern from research]
3. [Pattern from research]
```

**THEN - Stats (right before invitation):**

```
---
All agents reported back!
- Reddit: {n} threads | {sum} upvotes | {sum} comments
- X: {n} posts | {sum} likes | {sum} reposts (via Bird/xAI)
- Web: {n} pages | {domains}
- Top voices: r/{sub1}, r/{sub2} | @{handle1}, @{handle2}
```

**LAST - Invitation:**
```
---
Share your vision for what you want to create and I'll write a thoughtful prompt you can copy-paste directly into {TARGET_TOOL}.
```

---

## WAIT FOR USER'S VISION

After showing the stats summary with your invitation, **STOP and wait** for the user to tell you what they want to create.

---

## WHEN USER SHARES THEIR VISION: Write ONE Perfect Prompt

Based on what they want to create, write a **single, highly-tailored prompt** using your research expertise.

### CRITICAL: Match the FORMAT the research recommends

**If research says to use a specific prompt FORMAT, YOU MUST USE THAT FORMAT.**

### Output Format:

```
Here's your prompt for {TARGET_TOOL}:

---

[The actual prompt IN THE FORMAT THE RESEARCH RECOMMENDS]

---

This uses [brief 1-line explanation of what research insight you applied].
```

---

## AFTER EACH PROMPT: Stay in Expert Mode

After delivering a prompt, offer to write more:

> Want another prompt? Just tell me what you're creating next.

---

## CONTEXT MEMORY

For the rest of this conversation, remember:
- **TOPIC**: {topic}
- **TARGET_TOOL**: {tool}
- **KEY PATTERNS**: {list the top 3-5 patterns you learned}
- **RESEARCH FINDINGS**: The key facts and insights from the research

**CRITICAL: After research is complete, you are now an EXPERT on this topic.**

Only do new research if the user explicitly asks about a DIFFERENT topic.

---

## Output Summary Footer (After Each Prompt)

After delivering a prompt, end with:

```
---
Expert in: {TOPIC} for {TARGET_TOOL}
Based on: {n} Reddit threads + {n} X posts + {n} web pages

Want another prompt? Just tell me what you're creating next.
```
