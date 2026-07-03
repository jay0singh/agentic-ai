import os
import json
import base64
import requests
import urllib.parse
from typing import TypedDict, List, Dict, Any, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from core.retriever import retrieve
from core.memory import add_turn, get_history, format_history

load_dotenv()

# Langfuse tracing — only active when keys are configured.
_langfuse_handler = None
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    try:
        from langfuse.langchain import CallbackHandler
        _langfuse_handler = CallbackHandler()
        print("[Config] Langfuse tracing enabled.")
    except Exception as e:
        print(f"[Config] Langfuse init failed ({e}). Tracing disabled.")

# Both models run on Groq's free tier: a fast small model for routing/generation,
# a larger model for stricter answer evaluation.
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")

chat_model = ChatGroq(model=CHAT_MODEL, temperature=0)
judge_model = ChatGroq(model=JUDGE_MODEL, temperature=0)
print(f"[Config] Chat model: Groq ({CHAT_MODEL})")
print(f"[Config] Judge model: Groq ({JUDGE_MODEL})")


def _call_judge(prompt: str) -> str:
    """Invoke the judge LLM and return the raw text response."""
    return judge_model.invoke([HumanMessage(content=prompt)]).content


# ── State Definition ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    original_query: str
    history: str
    context: List[str]
    steps_taken: List[str]
    steps_remaining: List[str]
    next_node: str
    parameters: Dict[str, Any]
    response: Optional[str]
    retry_count: int
    judge_decision: str
    rewritten_query: Optional[str]
    judge_log: List[str]


# ── Helper REST Functions ──────────────────────────────────────────────────────

def web_search(query: str) -> str:
    """Perform web search using the Tavily REST API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set in environment."
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": 5},
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for r in data.get("results", []):
            results.append(f"Title: {r.get('title')}\nURL: {r.get('url')}\nSnippet: {r.get('content')}")
        return "\n\n---\n\n".join(results) if results else "No search results found."
    except Exception as e:
        return f"Error executing web search: {e}"


def github_read(operation: str, owner: str, repo: str, path: Optional[str] = None) -> str:
    """Retrieve repository details, README, or specific files via the GitHub REST API."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        if operation == "read_file":
            if not path:
                return "Error: File path is required for read_file operation."
            encoded_path = urllib.parse.quote(path,safe="/")
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{encoded_path}"
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json()
            content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
            return f"File '{path}' Content:\n{content}"

        elif operation == "read_readme":
            url = f"https://api.github.com/repos/{owner}/{repo}/readme"
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json()
            content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
            return f"README Content:\n{content}"

        elif operation == "list_issues":
            url = f"https://api.github.com/repos/{owner}/{repo}/issues"
            params = {"state": "open", "per_page": 5}
            res = requests.get(url, headers=headers, params=params, timeout=10)
            res.raise_for_status()
            issues = res.json()
            results = []
            for issue in issues:
                is_pr = "pull_request" in issue
                type_str = "PR" if is_pr else "Issue"
                results.append(f"{type_str} #{issue.get('number')}: {issue.get('title')} (State: {issue.get('state')})\nURL: {issue.get('html_url')}")
            return "\n\n".join(results) if results else "No open issues/PRs found."

        elif operation == "list_commits":
            url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            params = {"per_page": 5}
            if path:
                params["path"] = path
            res = requests.get(url, headers=headers, params=params, timeout=10)
            res.raise_for_status()
            commits = res.json()
            results = []
            for c in commits:
                commit = c.get("commit", {})
                author = commit.get("author", {})
                results.append(
                    f"SHA: {c.get('sha', '')[:7]}\n"
                    f"Message: {commit.get('message', '').splitlines()[0]}\n"
                    f"Author: {author.get('name')} <{author.get('email')}>\n"
                    f"Date: {author.get('date')}\n"
                    f"URL: {c.get('html_url')}"
                )
            return "\n\n".join(results) if results else "No commits found."

        else:
            return f"Error: Unsupported GitHub operation '{operation}'."
        
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code

        if status_code == 403:
            return (
                f"GitHub API Error: Rate limit exceeded or authentication failed "
                f"for repository '{owner}/{repo}'."
            )

        if status_code == 404:
            return (
            f"GitHub API Error: Repository '{owner}/{repo}' "
            f"or requested resource was not found."
        )

        return f"GitHub API Error ({status_code}): {e}"

    except requests.exceptions.Timeout:
        return "GitHub API Error: Request timed out."

    except requests.exceptions.RequestException as e:
        return f"GitHub API Error: {e}"

    except Exception as e:
        return f"Unexpected GitHub Error: {e}"


# ── Helper Parser & Intent Detection ───────────────────────────────────────────

def parse_json_response(content: str) -> dict:
    content = content.strip()
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    if start_idx != -1 and end_idx != -1:
        json_str = content[start_idx:end_idx+1]
        return json.loads(json_str)
    raise ValueError(f"No JSON object found in response: '{content}'")


def is_casual_query(query: str) -> bool:
    q = query.lower().strip().strip("?!.")
    
    # Collapse consecutive identical letters to handle spelling variations (e.g. heyy -> hey, hellooo -> helo)
    collapsed = ""
    for char in q:
        if not collapsed or char != collapsed[-1]:
            collapsed += char
            
    greetings = {
        "hey", "hello", "hi", "yo", "greetings", "good morning", "good afternoon", 
        "good evening", "howdy", "sup", "what's up", "help", "exit", "quit", "thanks", "thank you"
    }
    
    collapsed_greetings = {
        "hey", "helo", "hi", "yo", "gretings", "god morning", "god afternoon",
        "god evening", "howdy", "sup", "whats up", "help", "exit", "quit", "thanks", "thank you"
    }
    
    if q in greetings or collapsed in collapsed_greetings or len(q) <= 3:
        return True
        
    # Multi-word greeting check: e.g., "Hey bot", "Hello there"
    words = q.split()
    if words:
        first_word_collapsed = ""
        for char in words[0]:
            if not first_word_collapsed or char != first_word_collapsed[-1]:
                first_word_collapsed += char
        if (words[0] in greetings or first_word_collapsed in collapsed_greetings) and len(words) <= 3:
            return True
            
    return False


def resolve_followup(query: str, history_str: str) -> str:
    """Rewrite a follow-up question into a standalone one using the conversation
    history, so routing and retrieval work without the missing context."""
    prompt = f"""Given a conversation history and a follow-up user question, rewrite the question so it is fully standalone and understandable without the history.

Rules:
- Resolve pronouns and references ("it", "that", "there", "what about X") using the history.
- Keep the user's intent exactly — do not answer the question, only rewrite it.
- If the question is already standalone, return it unchanged.
- Output ONLY the rewritten question, nothing else.

CONVERSATION HISTORY:
{history_str}

FOLLOW-UP QUESTION: {query}

STANDALONE QUESTION:"""
    try:
        response = chat_model.invoke([HumanMessage(content=prompt)])
        resolved = response.content.strip().strip('"')
        if resolved:
            if resolved != query:
                print(f"  [Memory] Follow-up resolved to: '{resolved}'")
            return resolved
    except Exception as e:
        print(f"  [Memory] Follow-up resolution failed ({e}). Using original query.")
    return query


def classify_query(query: str) -> dict:
    """Classifies the user query to find the necessary tools and parameters."""
    prompt = f"""You are the Query Classifier. Analyze the user query and determine which tools are needed to answer it.

Available Tools:
- "vector_search": searches a PRIVATE internal database containing ONLY NimbusCart Inc. company documents (HR policies, employee benefits, NimbusDirect, NimbusMarket, account termination rules, shipping policies). Use ONLY for NimbusCart-specific queries.
- "web_search": searches the public internet for any general knowledge, sports, news, events, schedules, science, history, or external facts. Use for everything NOT related to NimbusCart's internal documents.
- "github_read": reads files, READMEs, or issues from a specific GitHub repository.

TOOL SELECTION RULES (follow strictly):
- Sports events, match results, schedules, scores → "web_search" ONLY (never "vector_search")
- News, current events, weather, science, history → "web_search" ONLY
- NimbusCart policies, NimbusDirect, NimbusMarket, employee benefits → "vector_search"
- GitHub repo content → "github_read"
- Mixed queries (e.g. NimbusCart policy + GitHub) → multiple tools

If "vector_search" or "web_search" is selected, extract:
- "search_query": A highly specific search query optimised for retrieval.

  CRITICAL RULES for search_query:
  1. PAST events ("last", "latest", "who won", "which team won"):
     → Include specific year/edition. E.g. "FIFA World Cup 2022 winner Argentina"
  2. FUTURE events ("next", "upcoming", "when is", "schedule", "fixture"):
     → Search for upcoming schedule. E.g. "FIFA World Cup 2026 next match schedule date"
     → NEVER use a past-event query for a future question.
  3. Company/policy queries: use exact brand names e.g. "NimbusDirect marketplace policy"
  4. Never be vague: "FIFA winner" is BAD. "FIFA World Cup 2022 winner" is GOOD.
  5. Match the tense of the user's question — past → past query, future → future query.

If "github_read" is selected, extract:
- "owner": repository owner (e.g. "langchain-ai")
- "repo": repository name (e.g. "langchain")
- "operation": "read_readme" | "list_issues" | "list_commits" | "read_file"
  - "read_readme": get the repository README
  - "list_issues": list open issues and pull requests
  - "list_commits": list recent commits (use for questions about recent changes, latest commit, commit history)
  - "read_file": read a specific file (requires "path")
- "path": (only for read_file, or optionally list_commits to filter by file path) the file path

Output ONLY a raw JSON object (no markdown, no explanation):
{{
  "tools": ["web_search"],
  "search_query": "targeted search keywords",
  "github_parameters": {{
    "owner": "owner_name",
    "repo": "repo_name",
    "operation": "operation_name",
    "path": "file_path"
  }}
}}

EXAMPLES (study these carefully):
- "Who won last FIFA?" → tools: ["web_search"], search_query: "FIFA World Cup 2022 winner"
- "When is the next FIFA match?" → tools: ["web_search"], search_query: "FIFA World Cup 2026 next match schedule"
- "Who won Super Bowl?" → tools: ["web_search"], search_query: "Super Bowl LVIII 2024 winner"
- "What is NimbusCart's leave policy?" → tools: ["vector_search"], search_query: "NimbusCart leave policy"
- "What is NimbusDirect?" → tools: ["vector_search"], search_query: "NimbusDirect"
- "Read the README for langchain-ai/langchain" → tools: ["github_read"], operation: "read_readme"
- "What are the open issues in langchain-ai/langchain?" → tools: ["github_read"], operation: "list_issues"
- "What is the most recent commit to langchain-ai/langchain?" → tools: ["github_read"], operation: "list_commits"
- "What is NimbusDirect and who won the World Cup?" → tools: ["vector_search", "web_search"]

USER QUERY: {query}
JSON OUTPUT:"""

    try:
        response = chat_model.invoke([HumanMessage(content=prompt)])
        return parse_json_response(response.content)
    except Exception as e:
        print(f"Classifier error: {e}. Defaulting to no tools.")
        return {"tools": [], "search_query": "", "github_parameters": {}}


# ── LangGraph Nodes ────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    """Static planner router: Decides step sequence on first run, then pops tools sequentially."""
    steps_taken = state.get("steps_taken", [])
    steps_remaining = state.get("steps_remaining")
    parameters = state.get("parameters", {})

    # 1. First run initialization
    if steps_remaining is None:
        if is_casual_query(state["query"]):
            print("  [Router] Casual query detected. Routing directly to generator.")
            return {
                "next_node": "generate",
                "steps_remaining": [],
                "steps_taken": ["generate"],
                "parameters": {}
            }
        
        # NimbusCart-specific queries always go to vector_search
        NIMBUSCART_KEYWORDS = {"nimbuscart", "nimbusdirect", "nimbusmarket"}
        if any(kw in state["query"].lower() for kw in NIMBUSCART_KEYWORDS):
            print("  [Router] NimbusCart query detected. Forcing vector_search.")
            return {
                "next_node": "vector_search",
                "steps_remaining": [],
                "steps_taken": steps_taken + ["vector_search"],
                "parameters": {"search_query": state["query"], "github": {}}
            }

        # Call Query Classifier
        classification = classify_query(state["query"])
        tools = classification.get("tools", []) or []
        search_query = classification.get("search_query") or state["query"]
        raw_github = classification.get("github_parameters") or {}
        github_params = {k: v for k, v in raw_github.items() if v is not None}

        # Build parameters
        parameters = {
            "search_query": search_query,
            "github": github_params
        }

        # Drop vector_search if the search_query is empty — no point in a semantic search with no keywords
        if not search_query.strip() and "vector_search" in tools:
            tools = [t for t in tools if t != "vector_search"]
            print("  [Router] Dropping vector_search (empty search_query).")

        # Classifier returned no tools for a substantive query — recover by defaulting to vector_search
        if not tools:
            print("  [Router] Classifier returned no tools. Defaulting to vector_search.")
            tools = ["vector_search"]
            if not parameters["search_query"].strip():
                parameters["search_query"] = state["query"]

        steps_remaining = tools
        print(f"  [Router] Planned tools to execute: {steps_remaining}")
        print(f"  [Router] Parameters extracted: {parameters}")

    # 2. Pop next tool if remaining
    if steps_remaining:
        next_tool = steps_remaining[0]
        remaining = steps_remaining[1:]
        print(f"  [Router] Routing to next node: '{next_tool}' (Remaining: {remaining})")
        return {
            "next_node": next_tool,
            "steps_remaining": remaining,
            "steps_taken": steps_taken + [next_tool],
            "parameters": parameters
        }
    
    # 3. Direct to generator if all tools completed
    print("  [Router] All planned steps completed. Routing to generator.")
    return {
        "next_node": "generate",
        "steps_remaining": [],
        "steps_taken": steps_taken + ["generate"]
    }


def vector_search_node(state: AgentState) -> dict:
    query = state["parameters"].get("search_query", state["query"])
    print(f"  [Node] Running Vector Search for: '{query}'...")
    chunks = retrieve(query, top_k=3)
    
    if chunks:
        source_content = f"--- [Vector Search Result for '{query}'] ---\n" + "\n---\n".join(chunks)
        new_context = state["context"] + [source_content]
    else:
        new_context = state["context"]
        
    return {
        "context": new_context,
        "next_node": "router"
    }


def web_search_node(state: AgentState) -> dict:
    query = state["parameters"].get("search_query", state["query"])
    print(f"  [Node] Running Tavily Web Search for: '{query}'...")
    results = web_search(query)
    
    if results and "No search results found." not in results and not results.startswith("Error"):
        source_content = f"--- [Web Search Result for '{query}'] ---\n{results}"
        new_context = state["context"] + [source_content]
    else:
        new_context = state["context"]
        
    return {
        "context": new_context,
        "next_node": "router"
    }


def github_read_node(state: AgentState) -> dict:
    params = state["parameters"].get("github", {})
    operation = params.get("operation", "read_readme")
    owner = params.get("owner")
    repo = params.get("repo")
    path = params.get("path")
    
    print(f"  [Node] Running GitHub Read: {operation} on {owner}/{repo} (path={path})...")
    
    if not owner or not repo:
        results = "Error: Repository owner and name must be specified in parameters."
    else:
        results = github_read(operation, owner, repo, path)
        
    if results and not results.startswith("Error") and not results.startswith("GitHub API Error"):
        source_content = f"--- [GitHub Result ({operation}) from {owner}/{repo}] ---\n{results}"
        new_context = state["context"] + [source_content]
    else:
        new_context = state["context"]
        
    return {
        "context": new_context,
        "next_node": "router"
    }


def generator_node(state: AgentState) -> dict:
    print("  [Node] Synthesizing final answer...")

    # Recent conversation turns (empty string when this is the first question)
    history_block = ""
    if state.get("history"):
        history_block = f"""
CONVERSATION SO FAR (for context — the question may refer back to it):
{state["history"]}
"""

    if not state.get("context"):
        # Respond directly if no context was retrieved (e.g. greetings, casual chat)
        prompt = f"""You are a helpful assistant. Respond to the user's input directly.
{history_block}
USER INPUT:
{state["query"]}

RESPONSE:"""
    elif "github_read" in state.get("steps_taken", []):
        # GitHub read: summarise the fetched content in response to the user's request
        context_str = "\n\n========================================\n\n".join(state["context"])
        prompt = f"""You are a helpful assistant. The user requested information from a GitHub repository.
Below is the content fetched from GitHub. Summarise it clearly and concisely in response to the user's request.
If the context contains a file or README, describe what the repository is, its purpose, and its key features.
If the context contains issues or PRs, list them clearly.

FETCHED GITHUB CONTENT:
{context_str}
{history_block}
USER REQUEST:
{state["query"]}

RESPONSE:"""
    else:
        # Context-based RAG prompt
        context_str = "\n\n========================================\n\n".join(state["context"])
        prompt = f"""You are a helpful assistant. Answer the user's question as accurately and helpfully as possible.

Context retrieved from search tools is provided below. Use it as follows:
- If the context DIRECTLY answers the question, use it and cite it.
- If the context is PARTIALLY relevant, use what applies and fill gaps with your own knowledge.
- If the context is IRRELEVANT (e.g. about a different event, time period, or topic), IGNORE it and answer from your own knowledge.
- For questions about UPCOMING events, schedules, or future dates: if context lacks this info, answer based on your training knowledge and clearly note it may not be current.
- ONLY respond with "I don't have enough information to answer that." for questions about PRIVATE INTERNAL data (like NimbusCart's internal policies) that genuinely cannot be answered without those documents.
- NEVER say you don't have information for publicly known facts, sports schedules, world events, or general knowledge questions.

ACCUMULATED CONTEXT:
{context_str}
{history_block}
USER QUESTION:
{state["query"]}

FINAL ANSWER:"""

    response = chat_model.invoke([HumanMessage(content=prompt)])
    return {
        "response": response.content.strip(),
        "next_node": "judge"
    }


# ── LangGraph Workflow compilation ──────────────────────────────────────────────

def route_next(state: AgentState) -> str:
    node = state.get("next_node", "generate")
    if node in ["vector_search", "web_search", "github_read", "generate"]:
        return node
    return "generate"


MAX_JUDGE_CONTEXT_CHARS = 4000


def _score_judge_decision(decision: str, reasoning: str) -> None:
    """Attach the judge's verdict to the active Langfuse trace, if tracing is enabled."""
    if not _langfuse_handler:
        return
    try:
        from langfuse import get_client
        get_client().score_current_trace(
            name="judge_decision",
            value=1 if decision == "accept" else 0,
            comment=reasoning,
        )
    except Exception as e:
        print(f"  [Judge] Langfuse scoring failed: {e}")


def judge_node(state: AgentState) -> dict:
    print("  [Judge] Evaluating answer...")
    print(f"  [Judge] Steps taken: {state.get('steps_taken', [])}")
    print(f"  [Judge] Context count: {len(state.get('context', []))}")
    print(f"  [Judge] Retry count: {state.get('retry_count', 0)}")

    steps_taken = state.get("steps_taken", [])
    judge_log = state.get("judge_log", [])
    context_str = "\n".join(state["context"])

    def _accept(reason: str) -> dict:
        print(f"  [Judge] {reason}")
        _score_judge_decision("accept", reason)
        return {
            "judge_decision": "accept",
            "steps_taken": steps_taken + ["judge"],
            "judge_log": judge_log + [f"ACCEPT: {reason}"]
        }

    # Tool failures should not trigger retries
    if (
        "GitHub API Error" in context_str
        or "Error executing web search" in context_str
        or "TAVILY_API_KEY" in context_str
        or "[TOOL_ERROR]" in context_str
    ):
        return _accept("Tool error detected. Accepting response.")

    # No context was retrieved — generator answered from training knowledge, nothing to judge against
    if not state.get("context"):
        return _accept("No context retrieved. Accepting response.")

    # Truncate to avoid token overflow on large web search results
    if len(context_str) > MAX_JUDGE_CONTEXT_CHARS:
        context_str = context_str[:MAX_JUDGE_CONTEXT_CHARS] + "\n...[context truncated]..."

    prompt = f"""You are a strict answer quality judge for a RAG (Retrieval-Augmented Generation) system.

ORIGINAL USER QUERY:
{state["original_query"]}

RETRIEVED CONTEXT:
{context_str}

GENERATED ANSWER:
{state["response"]}

EVALUATION STEPS — reason through each before deciding:

Step 1 — Relevance: Does the retrieved context relate to the user's query?
Step 2 — Grounding: Is the answer supported by the context, or does it contradict/ignore it?
Step 3 — Hallucination: Does the answer assert specific facts that are NOT present in the context and are unlikely to be common knowledge?
Step 4 — Completeness: Does the answer address the user's core question?
Step 5 — Decision: Based on steps 1–4, ACCEPT or RETRY?

ACCEPT when ANY of these apply:
- The answer is directly supported by the retrieved context.
- The answer correctly acknowledges that the context lacks sufficient information.
- The query is a greeting, small talk, or casual conversation.
- The answer is factually reasonable even if the context is only partially relevant.

RETRY only when ALL of these are true:
- Retrieval was attempted and context was returned.
- The answer ignores or contradicts the context without good reason, OR it fails to address the user's question.
- A more specific retrieval query would likely yield better context.

DO NOT retry because:
- The answer could be worded better.
- More information might exist elsewhere.
- The answer is cautious or hedged.

When rewriting the query for retry:
- Keep the user's original intent.
- Make it more specific and retrieval-optimised.
- Do NOT ask the user for clarification.

Return ONLY valid JSON (no markdown, no extra text):

ACCEPT format:
{{
  "reasoning": "brief explanation of why accepted",
  "decision": "accept"
}}

RETRY format:
{{
  "reasoning": "brief explanation of what is wrong with the answer",
  "decision": "retry",
  "rewritten_query": "improved retrieval query"
}}
"""

    try:
        raw = _call_judge(prompt)
        print(f"  [Judge] Raw response: {raw}")

        result = parse_json_response(raw)
        decision = result.get("decision", "accept")
        reasoning = result.get("reasoning", "")

        if decision == "retry":
            print(f"  [Judge] Retry requested. Reason: {reasoning}")
            print(f"  [Judge] Rewritten query: {result.get('rewritten_query')}")
            log_entry = f"RETRY: {reasoning} (rewritten query: \"{result.get('rewritten_query')}\")"
        else:
            print(f"  [Judge] Answer accepted. Reason: {reasoning}")
            log_entry = f"ACCEPT: {reasoning}"

        _score_judge_decision(decision, reasoning)

        return {
            "judge_decision": decision,
            "rewritten_query": result.get("rewritten_query"),
            "steps_taken": steps_taken + ["judge"],
            "judge_log": judge_log + [log_entry]
        }

    except Exception as e:
        print(f"  [Judge] Error parsing response: {e}")
        print("  [Judge] Defaulting to ACCEPT.")
        reason = f"Judge error ({e}). Defaulted to accept."
        _score_judge_decision("accept", reason)
        return {
            "judge_decision": "accept",
            "steps_taken": steps_taken + ["judge"],
            "judge_log": judge_log + [f"ACCEPT: {reason}"]
        }
    
def rewrite_node(state: AgentState) -> dict:
    steps_taken = state.get("steps_taken", [])

    if state["retry_count"] >= 3:
        print("  [Rewrite] Max retries reached.")
        return {"judge_decision": "accept", "steps_taken": steps_taken + ["rewrite"]}

    rewritten_query = state.get("rewritten_query") or state["query"]

    # Avoid an infinite loop if the judge returned an identical query
    if rewritten_query == state["query"]:
        print("  [Rewrite] Rewritten query is identical to original. Accepting.")
        return {"judge_decision": "accept", "steps_taken": steps_taken + ["rewrite"]}

    print(f"  [Rewrite] Retrying with query: {rewritten_query}")

    return {
        "query": rewritten_query,
        "context": [],
        "steps_remaining": None,
        "parameters": {},
        "retry_count": state["retry_count"] + 1,
        "response": None,
        "judge_decision": "accept",
        "rewritten_query": None,
        "steps_taken": steps_taken + ["rewrite"]
    }

def hitl_node(state: AgentState) -> dict:
    steps_taken = state.get("steps_taken", [])
    query = state.get("original_query", state["query"])
    retries = state.get("retry_count", 0)
    print(f"\n{'='*60}")
    print(f"  [HITL] Could not produce a satisfactory answer after {retries} retries.")
    print(f"  [HITL] Unanswered query: \"{query}\"")
    print(f"{'='*60}\n")
    return {"steps_taken": steps_taken + ["hitl"]}


def route_after_judge(state):
    if state["judge_decision"] == "retry":
        if state["retry_count"] < 3:
            return "rewrite"
        return "hitl"
    return "end"


workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("vector_search", vector_search_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("github_read", github_read_node)
workflow.add_node("generator", generator_node)
workflow.add_node("judge", judge_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("hitl", hitl_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_next,
    {
        "vector_search": "vector_search",
        "web_search": "web_search",
        "github_read": "github_read",
        "generate": "generator"
    }
)

workflow.add_edge("vector_search", "router")
workflow.add_edge("web_search", "router")
workflow.add_edge("github_read", "router")
workflow.add_edge("generator", "judge")

workflow.add_conditional_edges(
    "judge",
    route_after_judge,
    {
        "rewrite": "rewrite",
        "hitl": "hitl",
        "end": END
    }
)

workflow.add_edge("hitl", END)

workflow.add_edge(
    "rewrite",
    "router"
)

app = workflow.compile()


def run_orchestrator(query: str, session_id: Optional[str] = None, user_id: Optional[str] = None) -> dict:
    """Helper method to invoke the compiled LangGraph model."""
    history = get_history(session_id)
    history_str = format_history(history)

    # Rewrite follow-up questions ("what about X?") into standalone ones so the
    # router and retrieval see the full intent, not a fragment.
    resolved_query = query
    if history and not is_casual_query(query):
        resolved_query = resolve_followup(query, history_str)

    initial_state = {
        "query": resolved_query,
        "original_query": resolved_query,
        "history": history_str,
        "context": [],
        "steps_taken": [],
        "steps_remaining": None,
        "next_node": "router",
        "parameters": {},
        "response": None,
        "retry_count": 0,
        "judge_decision": "accept",
        "rewritten_query": None,
        "judge_log": [],
    }
    if not _langfuse_handler:
        state = app.invoke(initial_state)
        add_turn(session_id, query, state.get("response"))
        return state

    from langfuse import get_client, propagate_attributes

    config = {"callbacks": [_langfuse_handler]}
    attrs = {"trace_name": "rag-chat", "tags": ["rag-demo"]}
    if session_id:
        attrs["session_id"] = session_id
    if user_id:
        attrs["user_id"] = user_id

    # Wrap the graph invocation in a root span so the trace input/output show the
    # user's question and final answer instead of the full AgentState dict.
    trace_input = {"question": query}
    if resolved_query != query:
        trace_input["resolved_question"] = resolved_query

    with propagate_attributes(**attrs):
        with get_client().start_as_current_observation(as_type="span", name="rag-chat") as span:
            span.set_trace_io(input=trace_input)
            state = app.invoke(initial_state, config=config)
            span.set_trace_io(output={"answer": state.get("response")})

    add_turn(session_id, query, state.get("response"))
    return state