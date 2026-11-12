# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.1,<0.8.0", "ipython",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

"""Twitch chat filter, from https://console.typesafe.ai/docs/cookbooks/twitch_chat_filter.

Needs no API key: it replays this cookbook's saved answers. To call the API,
set TYPESAFE_API_KEY and DOWNLOAD_CACHE = False.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import urllib.request
from pathlib import Path
from time import perf_counter

from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import Choice, Noul, NoulCriteria, TypeSafeClient

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live

client = TypeSafeClient(
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),  # cached re-renders need no key
    base_url=os.environ.get("TYPESAFE_ENDPOINT"),
    timeout=60.0,
)
CACHE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

EXAMPLES_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/twitch_chat_filter"
)
CACHE = CACHE_DIR / "twitch_chat_filter_cache.json"
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE and not CACHE.exists():
    urllib.request.urlretrieve(f"{EXAMPLES_URL}/{CACHE.name}", CACHE)
json_cache = JsonCache(CACHE)

STREAM = {"title": "Elden Ring first playthrough! No spoilers please", "game": "Elden Ring"}

CHAT = [
    ("night-owl-22", "hi chat, first time catching a stream live"),
    ("ggwp-andy", "PogChamp PogChamp PogChamp"),
    ("mira-plays", "how many hours have you played so far?"),
    ("tarnished-tom", "this area looks so good at night"),
    ("backseat-bert", "go left, there's a site of grace behind the tree"),
    ("follow4follow-bot", "Want more viewers? Buy 1000 followers for $5 at cheapviewz dot example"),
    ("lore-hound", "wait till you find out Ranni planned the Night of Black Knives"),
    ("kappa-kev", "LUL LUL LUL"),
    ("sunny-skies", "are you going to do a no-hit run after this?"),
    ("rage-ryan", "you're genuinely the worst player i've ever seen, uninstall trash"),
    ("cozy-cat", "the music in this area is so good"),
    ("backseat-bert", "use your flask NOW, why are you not healing"),
    ("mira-plays", "W streamer"),
    ("ggwp-andy", "GG that was insane"),
    ("quiet-quinn", "anyone else watching from Brazil?"),
    ("tarnished-tom", "what build are you going for?"),
    ("lore-hound", "the ending where you become Elden Lord with Ranni is the best one"),
    ("kappa-kev", "chat is wild today lol"),
    ("backseat-bert", "the boss is weak to fire, equip the torch before you go in"),
    ("rage-ryan", "nobody cares about your opinion, go back to your basement"),
    ("cozy-cat", "you're so bad at this boss lmao, love you tho"),
    ("sunny-skies", "who is the final boss?"),
    ("ggwp-andy", "LETS GOOOO"),
    ("night-owl-22", "i just got this game too, it's so hard"),
    ("promo-pete", "check out my channel, I stream every day, follow me!"),
    ("quiet-quinn", "dodge rolling through that was clean"),
    ("mira-plays", "is the DLC worth it?"),
    ("cozy-cat", "your character looks so cool"),
    ("night-owl-22", "this boss took me 40 tries lol"),
]

print(f"{len(CHAT)} messages. What one request sends as its state:")
print({"stream": STREAM, "message": {"user": CHAT[4][0], "text": CHAT[4][1]}})

QUESTIONS = {
    "category": Choice(
        instructions="Which category best describes the main purpose of `message.text`?",
        criteria={
            "question": "A genuine request for information from the streamer or chat, "
            "not gameplay instructions phrased as a question or a promotional pitch",
            "hype": "Cheering, reactions, or repeated emotes (PogChamp, LUL, GG, W, "
            "lets gooo); repetition alone does not make cheering spam",
            "conversation": "Ordinary chatter, opinions, replies, useful shared links, "
            "or any message whose main purpose does not fit the other categories",
            "backseat": "Unrequested gameplay advice telling the streamer what to do, "
            "including instructions phrased as a question; not a request for information",
            "spam": "Unsolicited advertising, self-promotion, follower selling, or "
            "promotional links; not ordinary cheering, repeated emotes, or a useful link",
        },
    ),
    "toxic": Noul(
        instructions="Is this message insulting, hateful or harassing towards someone?",
        criteria=NoulCriteria(
            true="Insults, slurs, threats or personal attacks on the streamer or another viewer",
            false="Friendly, neutral, or light banter with no target",
        ),
    ),
    "spoiler": Noul(
        instructions="Does this message reveal a plot point or ending of the game being played?"
    ),
}

def classify(user: str, text: str, stream: dict | None = None) -> dict:
    state = {
        "stream": STREAM if stream is None else stream,
        "message": {"user": user, "text": text},
    }
    questions = {key: q.model_dump(mode="json", exclude_none=True) for key, q in QUESTIONS.items()}
    return _ask(state, questions, model=TYPESAFE_MODEL)


@json_cache
def _ask(state: dict, questions: dict, model: str) -> dict:
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("Set TYPESAFE_API_KEY to send requests to TypeSafe.")
    started = perf_counter()
    response = client.system_one(state=state, questions=questions, model=model)
    category = response.answers["category"]
    return {
        "category": category.choice,
        "probabilities": dict(category.probabilities),
        "toxic": response.answers["toxic"].noul,
        "spoiler": response.answers["spoiler"].noul,
        "seconds": round(perf_counter() - started, 2),
    }


def show(user: str, text: str) -> None:
    result = classify(user, text)
    print(f"{user}: {text}")
    for label, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:<13}{p:5.2f}  {'#' * round(p * 30)}".rstrip())
    print(f"  {'toxic':<13}{result['toxic']:5.2f}")
    print(f"  {'spoiler':<13}{result['spoiler']:5.2f}\n")


show("backseat-bert", "go left, there's a site of grace behind the tree")
show("lore-hound", "wait till you find out Ranni planned the Night of Black Knives")
show("cozy-cat", "you're so bad at this boss lmao, love you tho")

RULES = {
    "question": "highlight",
    "hype": "show",
    "conversation": "show",
    "backseat": "hide",
    "spam": "hide",
}
TOXIC_AT = 0.7  # hide at or above this chance of toxic
SPOILER_AT = 0.7  # blur at or above this chance of a spoiler


def decide(result: dict, rules: dict = RULES) -> str:
    if result["toxic"] >= TOXIC_AT:
        return "hide (toxic)"
    action = rules[result["category"]]
    if action == "hide":
        return f"hide ({result['category']})"
    if result["spoiler"] >= SPOILER_AT:
        return "blur (spoiler)"
    return action

with ThreadPoolExecutor(max_workers=8) as pool:  # 29 requests, one per message, 8 at once
    results = list(pool.map(lambda message: classify(*message), CHAT))

counts: dict[str, int] = {}
for (user, text), result in zip(CHAT, results):
    action = decide(result)
    counts[action] = counts.get(action, 0) + 1
    if action.startswith("hide"):
        line = f"   · hidden, {action[6:-1]}"
    elif action == "blur (spoiler)":
        line = f"   {user}: ░░░░░░░░ spoiler"
    else:
        mark = {"highlight": ">>", "dim": "..", "show": "  "}[action]
        line = f"{mark} {user}: {text}"
    print(f"{line[:78]:<80}{result['category']}")

print()
for action, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"{action:<20}{n:>3}")

QUIET = RULES | {"hype": "hide", "conversation": "dim"}

changed = 0
for (user, text), result in zip(CHAT, results):
    before, after = decide(result), decide(result, QUIET)
    if before != after:
        changed += 1
        print(f"{user + ': ' + text:<62.62}  {before:<10} -> {after}")
print(f"\n{changed} messages changed, 0 new requests")

playground_link = make_playground_link(
    {"stream": STREAM, "message": {"user": CHAT[4][0], "text": CHAT[4][1]}},
    QUESTIONS,
)
# Print a usable URL in a script; render a clickable link in a notebook.
if __name__ == "__main__" and "__file__" in globals():
    print(f"Open in the TypeSafe playground:\n{playground_link}")
else:
    display(Markdown(f"🔗 [Open in the TypeSafe playground]({playground_link})"))
