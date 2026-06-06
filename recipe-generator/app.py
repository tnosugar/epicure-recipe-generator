"""
app.py - Flask backend for the epicure-cooc recipe generator.

Pipeline per chat message:
  1. model_layer parses the prompt and builds an ingredient brief from epicure-cooc
  2. the brief + conversation history + current recipe are sent to Claude
  3. Claude returns a structured recipe (JSON), which the frontend renders

Run:
  export ANTHROPIC_API_KEY=sk-...
  pip install -r requirements.txt
  python app.py            # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque

from flask import Flask, jsonify, request, send_from_directory

from model_layer import RecipeModel

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
HERE = os.path.dirname(os.path.abspath(__file__))

# Abuse guards for public deployments: per-IP burst limit + global daily cap.
# In-memory; resets on restart. Fine for a single-instance prototype.
RATE_PER_IP_PER_MIN = int(os.environ.get("RATE_PER_IP_PER_MIN", "6"))
RATE_DAILY_CAP = int(os.environ.get("RATE_DAILY_CAP", "100"))
_ip_hits: dict[str, deque] = defaultdict(deque)
_daily = {"day": "", "count": 0}


def _rate_limited(ip: str) -> str | None:
    now = time.time()
    today = time.strftime("%Y-%m-%d")
    if _daily["day"] != today:
        _daily["day"], _daily["count"] = today, 0
    if _daily["count"] >= RATE_DAILY_CAP:
        return "Daily recipe budget reached. Try again tomorrow."
    hits = _ip_hits[ip]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= RATE_PER_IP_PER_MIN:
        return "Too many requests - wait a minute and try again."
    hits.append(now)
    _daily["count"] += 1
    return None

app = Flask(__name__, static_folder=None)

# Load the embedding once at startup (a few MB; fast).
print("Loading epicure-cooc ...")
MODEL = RecipeModel()
print(f"Loaded. Cuisines: {sorted(MODEL.cuisine_poles)}")


SYSTEM_PROMPT = """You are the recipe-writing engine of a cooking app.

An ingredient-pairing model (epicure-cooc, trained on 4.14M recipes) has already
analysed the user's request and produced an INGREDIENT BRIEF: seed ingredients
the user named, an optional target cuisine, dietary constraints, and a ranked
list of co-occurring candidate ingredients with similarity scores.

Your job: write ONE coherent, genuinely cookable recipe that is built AROUND the
brief. Lean on the brief's candidate ingredients (you don't have to use all of
them, and you may add obvious staples like oil, salt, water), honour every
dietary constraint strictly, and respect the cuisine if given.

When given a CURRENT RECIPE and a refinement request, EDIT that recipe rather
than starting over - keep what still works.

Respond with ONLY a JSON object, no prose around it, in exactly this shape:
{
  "title": "short dish name",
  "description": "1-2 sentence intro",
  "servings": 4,
  "time": "e.g. 45 min",
  "ingredients": [{"item": "beef chuck", "quantity": "500 g"}],
  "steps": ["step 1", "step 2"],
  "pairing_notes": "1-3 sentences on how the model's suggested pairings shaped this dish",
  "chat_reply": "a short, friendly one-line reply to show in the chat pane"
}
"""


def _build_user_message(brief: dict, current_recipe, latest: str) -> str:
    parts = [f"USER REQUEST:\n{latest}\n"]
    parts.append("INGREDIENT BRIEF (from epicure-cooc):")
    parts.append(json.dumps({
        "seeds": brief["seeds"],
        "cuisine": brief["cuisine"],
        "constraints": brief["constraints"],
        "candidates": [{"ingredient": c["name"].replace("_", " "),
                        "score": c["score"], "why": c["source"]}
                       for c in brief["candidates"]],
    }, indent=2))
    if current_recipe:
        parts.append("\nCURRENT RECIPE (edit this for refinements):")
        parts.append(json.dumps(current_recipe, indent=2))
    return "\n".join(parts)


def _call_claude(history, system, user_msg):
    """Call the Claude API. Returns parsed recipe dict."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msgs = []
    for h in history[:-1]:  # prior turns for context
        role = "user" if h.get("role") == "user" else "assistant"
        msgs.append({"role": role, "content": h.get("content", "")})
    msgs.append({"role": "user", "content": user_msg})

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system,
        messages=msgs,
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return _parse_json(text)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()
    blocked = _rate_limited(ip)
    if blocked:
        return jsonify({"error": blocked}), 429

    data = request.get_json(force=True)
    messages = data.get("messages", [])
    current_recipe = data.get("current_recipe")
    if not messages:
        return jsonify({"error": "no messages"}), 400

    latest = messages[-1].get("content", "")
    parsed = MODEL.parse(latest)
    # On a refinement with no new ingredients, seed from the current recipe.
    if not parsed["ingredients"] and current_recipe:
        reseed = " ".join(i.get("item", "") for i in current_recipe.get("ingredients", []))
        parsed["ingredients"] = MODEL.parse(reseed)["ingredients"][:4]
    brief = MODEL.build_brief(parsed)

    user_msg = _build_user_message(brief, current_recipe, latest)
    try:
        recipe = _call_claude(messages, SYSTEM_PROMPT, user_msg)
    except Exception as e:  # surface a clean error to the UI
        return jsonify({"error": f"recipe generation failed: {e}", "brief": brief}), 502

    chat_reply = recipe.pop("chat_reply", None) or "Here's your recipe."
    return jsonify({"recipe": recipe, "brief": brief, "assistant_message": chat_reply})


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "model": CLAUDE_MODEL,
                    "has_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
                    "cuisines": sorted(MODEL.cuisine_poles)})


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is not set - recipe writing will fail.")
    port = int(os.environ.get("PORT", "5000"))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    app.run(host=host, port=port, debug=not bool(os.environ.get("PORT")))
