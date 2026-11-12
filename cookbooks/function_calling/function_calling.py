# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.0", "ipython", "polars", "matplotlib", "numpy",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

"""Function calling, from https://console.typesafe.ai/docs/cookbooks/function_calling.

Needs no API key: it replays this cookbook's saved answers. To call the API,
set TYPESAFE_API_KEY and DOWNLOAD_CACHE = False.
"""

import json
import os
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import SystemOneResponse, TypeSafeClient

EXAMPLES = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/function_calling"
)
for name in ["dispatch.py", "trader.py", "spec.json", "bars.parquet"]:
    if not Path(name).exists():
        urllib.request.urlretrieve(f"{EXAMPLES}/{name}", name)

from dispatch import ROUTE, Dispatcher, closed_sets  # noqa: E402  (downloaded above)
from trader import TOOLS, chart_theme, load  # noqa: E402


def show(draw) -> None:
    """Display the chart ``draw()`` makes. The docs build sets COOKSAFE_RENDER and also gets
    a dark copy."""
    for theme in ("light", "dark") if os.environ.get("COOKSAFE_RENDER") else ("light",):
        with chart_theme(theme):
            figure = draw()
        display(figure, metadata={"theme": theme})
        plt.close(figure)


print(f"{len(TOOLS)} functions over {load().height:,} one-minute bars")

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
typesafe = TypeSafeClient(
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),  # keyless runs replay the cache
    base_url=os.environ.get("TYPESAFE_ENDPOINT"),
    timeout=120.0,
)

CACHE = Path("function_calling_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE and not CACHE.exists():
    urllib.request.urlretrieve(f"{EXAMPLES}/{CACHE.name}", CACHE)
json_cache = JsonCache(CACHE)


@json_cache
def _system_one(state: str, questions: str, model: str) -> dict:
    response = typesafe.system_one(state=state, questions=json.loads(questions), model=model)
    return {
        "answers": {k: a.model_dump(mode="json") for k, a in response.answers.items()},
        "usage": response.usage.model_dump(mode="json"),
    }


class Client:
    """The `system_one` the dispatcher calls, with every answer cached in CACHE."""

    def system_one(self, state, questions: dict, model: str) -> SystemOneResponse:
        raw = _system_one(str(state), json.dumps(questions, sort_keys=True), model=model)
        return SystemOneResponse.model_validate({**raw, "model": model})


client = Client()

for name, fn in TOOLS.items():
    shapes = closed_sets(fn)
    print(
        f"  {name:<20}{len(shapes)}  "
        + ", ".join(f"{a}:{s}" for a, (s, _) in shapes.items())
    )
print(
    f"\n{sum(len(closed_sets(fn)) for fn in TOOLS.values())} fillable arguments in total"
)

SPEC = json.loads(Path("spec.json").read_text())
for argument in ("style", "moving_average"):
    print(
        json.dumps(
            {argument: SPEC["functions"]["plot_price"]["arguments"][argument]}, indent=2
        )
    )

assistant = Dispatcher(SPEC, TOOLS, client, model=TYPESAFE_MODEL)
print(f"{len(assistant.questions)} questions per command, among them:")
for qid in (
    "__tool__",
    "plot_price.style",
    "plot_price.style?",
    "compare_returns.symbols.NVDA",
):
    question = assistant.questions[qid]
    print(f"  {qid:<30}{question['type']:<8}{str(question['instructions'])[:64]}")

COMMANDS = [
    "show nvda 1h",
    "plot rolling correlation between nvda and spy for the past month",
    "when during the day does nvda trade the most",
    "what moved today",
    "what tickers do you have",
    "how did the market do this week",
    "candles for tesla with a 20 period moving average",
    "compare nvda amd and msft over the past three months",
    "how volatile is tsla",
    "biggest losers today",
    "worst drawdown for nvda this quarter, and chart it please",
    "spy stats for the last month",
    "show me apple daily with volume",
    "is amd tracking nvidia lately",
]

CALLS = {command: assistant(command) for command in COMMANDS}
for command, call in CALLS.items():
    print(f'  "{command}"')
    print(
        f"      {str(call):<66}confidence {call.confidence:.2f}"
        f"   tool {call.tool.probability:.2f}"
    )

for command in (
    "plot rolling correlation between nvda and spy for the past month",
    "compare nvda amd and msft over the past three months",
    "when during the day does nvda trade the most",
):
    print(f'"{command}"  ->  {CALLS[command]}')
    show(CALLS[command].run)

for command in ("how did the market do this week", "biggest losers today"):
    print(f'"{command}"  ->  {CALLS[command]}')
    print(CALLS[command].run(), "\n")

call = CALLS["is amd tracking nvidia lately"]
print(f'"is amd tracking nvidia lately"  ->  {call}   confidence {call.confidence:.2f}')
for name, argument in call.arguments.items():
    top = sorted(argument.distribution.items(), key=lambda kv: -kv[1])[:3]
    shown = "omitted, default stands" if argument.omitted else repr(argument.value)
    print(
        f"  {name:<12}{shown:<26}p {argument.probability:.2f}   "
        + "  ".join(f"{k} {v:.2f}" for k, v in top)
    )
print(f"  weakest argument: {call.weakest().name}")

COMMAND = "plot rolling correlation between nvda and spy for the past month"
picked = CALLS[COMMAND]
playground_link = make_playground_link(
    COMMAND,
    {ROUTE: assistant.questions[ROUTE]}
    | {q: v for q, v in assistant.questions.items() if q.startswith(f"{picked.name}.")},
)
display(
    Markdown(
        f"🔗 [Open the command and its questions in the TypeSafe playground]({playground_link})"
    )
)
