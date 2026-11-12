# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.1", "ipython",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

"""Shopping cart, from https://console.typesafe.ai/docs/cookbooks/cart_from_primitives.

Needs no API key: it replays this cookbook's saved answers. To call the API,
set TYPESAFE_API_KEY and DOWNLOAD_CACHE = False.
"""

import os
import urllib.request
from pathlib import Path

from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import Choice, TypeSafeClient

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
NOTHING_ELSE = "nothing_else"  # the sentinel option that ends the item loop
MAX_ITEM_REQUESTS = 6  # a hard stop, so a bad answer can never loop forever

client = TypeSafeClient(
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),  # cached re-renders need no key
    model=TYPESAFE_MODEL,
    timeout=30.0,
)

EXAMPLES_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/cart_from_primitives"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        urllib.request.urlretrieve(f"{EXAMPLES_URL}/{name}", path)
    return path


CACHE = Path("cart_from_primitives_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)


def fmt_confidence(confidence: float | None) -> str:
    """A confidence for printing. The API can return None, so never format it blind."""
    return f"{confidence:.2f}" if confidence is not None else "n/a"

MENU = {
    "latte": {
        "description": "Espresso with steamed milk. Also ordered as a flat white or a milky coffee.",
        "fields": {
            "size": ["small", "medium", "large"],
            "milk": ["whole", "oat", "almond"],
        },
    },
    "filter_coffee": {
        "description": "Plain brewed drip coffee, sometimes ordered as a black coffee.",
        "fields": {},
    },
    "blueberry_muffin": {
        "description": "A blueberry muffin.",
        "fields": {},
    },
    "almond_croissant": {
        "description": "An almond croissant, sometimes ordered as a pastry.",
        "fields": {},
    },
}

ORDER = "Can I get a large oat milk latte and a blueberry muffin please"

def item_question(remaining: dict[str, dict], found: list[str]) -> Choice:
    """Build one request's question from the items not found yet."""
    already = ", ".join(found) if found else "nothing yet"
    criteria = {name: item["description"] for name, item in remaining.items()}
    criteria[NOTHING_ELSE] = (
        "The order mentions no menu item other than the ones already accounted for."
    )
    return Choice(
        instructions=(
            f"This is a cafe order. Items already accounted for: {already}. "
            "Which remaining menu item does the order mention? If the order mentions more "
            "than one remaining item, choose the one mentioned first."
        ),
        criteria=criteria,
    )


@json_cache
def ask_item(order: str, question: dict, model: str) -> dict:
    """One item request. The question arrives as its API JSON, which also keys the cache."""
    answer = client.system_one(
        state=order, questions={"item": question}, model=model
    ).choices["item"]
    return {"choice": answer.choice, "confidence": answer.confidence}


def find_items(order: str) -> list[dict]:
    remaining = list(MENU)
    found: list[dict] = []
    item_requests = 0
    for _ in range(MAX_ITEM_REQUESTS):
        if not remaining:
            break
        question = item_question(
            {name: MENU[name] for name in remaining},
            [item["name"] for item in found],
        )
        result = ask_item(
            order, question.model_dump(mode="json", exclude_none=True), model=TYPESAFE_MODEL
        )
        item_requests += 1
        print(
            f"item request {item_requests}: {len(remaining)} items + sentinel "
            f"-> {result['choice']} (confidence {fmt_confidence(result['confidence'])})"
        )
        if result["choice"] == NOTHING_ELSE:
            break
        remaining.remove(result["choice"])
        found.append({"name": result["choice"], "confidence": result["confidence"]})
    return found


items = find_items(ORDER)
print(f"\nfound {len(items)} items: {[item['name'] for item in items]}")

UNSPECIFIED = "unspecified"


def field_questions(item_name: str) -> dict[str, Choice]:
    questions = {}
    for field, values in MENU[item_name]["fields"].items():
        criteria = {value: None for value in values}
        criteria[UNSPECIFIED] = (
            f"The order does not say which {field} for the {item_name}."
        )
        questions[field] = Choice(
            instructions=f"Which {field} did the order ask for, for the {item_name}?",
            criteria=criteria,
        )
    return questions


@json_cache
def ask_fields(order: str, questions: dict[str, dict], model: str) -> dict:
    """One field request. The questions arrive as their API JSON, which also keys the cache."""
    response = client.system_one(state=order, questions=questions, model=model)
    return {
        field: {"choice": answer.choice, "confidence": answer.confidence}
        for field, answer in response.choices.items()
    }


def fill_fields(order: str, item_name: str) -> dict:
    if not MENU[item_name]["fields"]:
        return {}
    questions = {
        field: question.model_dump(mode="json", exclude_none=True)
        for field, question in field_questions(item_name).items()
    }
    return ask_fields(order, questions, model=TYPESAFE_MODEL)


cart = []
for item in items:
    fields = fill_fields(ORDER, item["name"])
    cart.append(
        {"name": item["name"], "confidence": item["confidence"], "fields": fields}
    )

print(f'order: "{ORDER}"\n')
for line in cart:
    print(f"{line['name']}  (item confidence {fmt_confidence(line['confidence'])})")
    for field, answer in line["fields"].items():
        conf = fmt_confidence(answer["confidence"])
        print(f"    {field:<6} {answer['choice']:<12} confidence {conf}")

playground_link = make_playground_link(ORDER, {"item": item_question(MENU, [])})
display(
    Markdown(
        f"🔗 [Open the first item-finding request in the TypeSafe playground]({playground_link})"
    )
)
