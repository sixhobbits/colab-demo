# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.0", "ipython",
#                 "cooksafe>=0.3.0,<0.4.0"]
# ///

"""Parallel questions, from https://console.typesafe.ai/docs/cookbooks/parallel_questions.

Set TYPESAFE_API_KEY first. It replays this cookbook's saved answers; to call
the API instead, set DOWNLOAD_CACHE = False.
"""

import json
import os
import urllib.request
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, TypeSafeClient

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
PRICE = (
    0.042,
    0.00,
)  # $ per 1M tokens (input, output); TypeSafe jev-1.14.0 as of 2026-11, see README
RUNS = 5  # repeats per batching strategy, to estimate each answer's run-to-run std dev
client = TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"], timeout=120.0)

EXAMPLES_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/parallel_questions"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        urllib.request.urlretrieve(f"{EXAMPLES_URL}/{name}", path)
    return path


CACHE = Path("parallel_questions_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)

WIKIPEDIA_REVISION = 1363040264  # "General Data Protection Regulation", as of 2026-07


@json_cache
def fetch_article(revision_id: int) -> str:
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&format=json"
        f"&prop=extracts&explaintext=1&revids={revision_id}"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "typesafe-cookbook/1.0"}
    )
    with urllib.request.urlopen(request) as response:
        pages = json.loads(response.read())["query"]["pages"]
    return next(iter(pages.values()))["extract"]


DOCUMENT = {
    "source": f"https://en.wikipedia.org/?oldid={WIKIPEDIA_REVISION}",
    "text": fetch_article(WIKIPEDIA_REVISION),
}
print(f"{len(DOCUMENT['text']):,} characters")
display(Markdown(f"📄 [Read the pinned Wikipedia revision]({DOCUMENT['source']})"))

QUESTIONS = {
    "breach_72h": Noul(
        instructions="Must a personal data breach be reported to the supervisory authority within 72 hours?"
    ),
    "applies_non_eu": Noul(
        instructions="Does the regulation apply to organisations established outside the EU that offer goods or services to people in the EU?"
    ),
    "dpo_all_orgs": Noul(
        instructions="Must every organisation appoint a Data Protection Officer, regardless of what data it processes?"
    ),
    "pre_ticked_consent": Noul(
        instructions="Can valid consent be obtained through pre-ticked boxes or inactivity?"
    ),
    "right_erasure": Noul(
        instructions="Does the regulation grant individuals a right to erasure of their personal data?"
    ),
    "data_portability": Noul(
        instructions="Does the regulation include a right to data portability?"
    ),
    "us_federal_law": Noul(instructions="Is the GDPR a United States federal law?"),
    "criminal_penalties": Noul(
        instructions="Does the GDPR itself impose criminal penalties such as imprisonment?"
    ),
    "instrument_type": Choice(
        instructions="What kind of EU legal instrument is the GDPR?",
        criteria={
            "Regulation": "Directly binding law in all member states, no national implementation needed.",
            "Directive": "Sets goals that member states implement through national law.",
            "Treaty": "An international treaty between states.",
            "Recommendation": "Non-binding guidance.",
        },
    ),
    "max_fine": Choice(
        instructions="What is the maximum administrative fine for the most serious infringements?",
        criteria={
            "TwentyM_or_4pct": "Up to EUR 20 million or 4% of annual worldwide turnover, whichever is greater.",
            "TenM_or_2pct": "Up to EUR 10 million or 2% of annual worldwide turnover, whichever is greater.",
            "FixedCap": "A fixed amount not tied to turnover.",
            "NoFines": "The GDPR provides no administrative fines.",
        },
    ),
    "individual_rights": Score(
        instructions="How strong are the rights the GDPR grants to individuals over their data?",
        criteria=[
            "None: individuals get no rights over their data.",
            "Weak: a right to be informed, but little control.",
            "Moderate: access and correction rights, but limited means to act on them.",
            "Strong: access, erasure, portability, and objection rights, with enforcement behind them.",
        ],
    ),
    "penalty_severity": Score(
        instructions="How severe are the penalties the GDPR provides for non-compliance?",
        criteria=[
            "None: no penalties of any kind.",
            "Symbolic: small fixed fines unlikely to change behavior.",
            "Substantial: fines large enough to matter to most companies.",
            "Severe: fines scaled to global revenue, material even to the largest companies.",
        ],
    ),
    "compliance_burden": Score(
        instructions="How heavy is the compliance burden the GDPR places on organisations?",
        criteria=[
            "Negligible: no meaningful obligations.",
            "Light: a few notices and disclosures.",
            "Moderate: documented processes and some dedicated roles for larger processors.",
            "Heavy: records, impact assessments, officers, and breach procedures for many organisations.",
            "Extreme: obligations so demanding that ordinary organisations cannot fully comply.",
        ],
    ),
}
N = len(QUESTIONS)
METRIC = {  # question type -> the one number we track per answer
    Noul: "p(yes)",
    Choice: "max prob",
    Score: "normalized score",
}

@json_cache
def ask(document_json: str, questions_json: str, run: int, model: str):
    """One TypeSafe call -> ({key: tracked metric}, input_tokens, output_tokens, latency_s);
    ``run`` only forces a distinct live call per repeat."""
    started = perf_counter()
    questions = json.loads(questions_json)
    response = client.system_one(
        state={"article": json.loads(document_json)},
        questions=questions,
        model=model,
    )
    values = {}
    for key, question in questions.items():
        answer = response.answers[key]
        if isinstance(answer, NoulAnswer):
            values[key] = answer.noul
        elif isinstance(answer, ChoiceAnswer):
            values[key] = max(answer.probabilities.values())
        else:
            values[key] = answer.score / (len(question["criteria"]) - 1)
    return (
        values,
        response.usage.input_tokens,
        response.usage.output_tokens,
        perf_counter() - started,
    )


def ask_keys(keys: tuple[str, ...], run: int):
    """Ask the questions named by ``keys``, sending their full text."""
    questions = {key: QUESTIONS[key].model_dump(mode="json", exclude_none=True) for key in keys}
    return ask(json.dumps(DOCUMENT), json.dumps(questions), run, model=TYPESAFE_MODEL)


def priced(result):
    """({key: metric}, in_tokens, out_tokens, latency) -> ({key: metric}, cost_usd, latency)."""
    values, input_tokens, output_tokens, latency = result
    return values, input_tokens / 1e6 * PRICE[0] + output_tokens / 1e6 * PRICE[1], latency


# Price after cache retrieval, so a price change needs no new calls.
batched = [
    priced(ask_keys(tuple(QUESTIONS), run)) for run in range(RUNS)
]  # all N in one call, x RUNS
singles = [
    {key: priced(ask_keys((key,), run)) for key in QUESTIONS} for run in range(RUNS)
]  # N x 1, x RUNS

print(
    f"{'question':<22}{'metric':<18}{'batched mean':>13}{'single mean':>12}"
    f"{'batched std':>13}{'single std':>12}"
)
for key, question in QUESTIONS.items():
    batched_values = [values[key] for values, _cost, _latency in batched]
    single_values = [singles[run][key][0][key] for run in range(RUNS)]
    print(
        f"{key:<22}{METRIC[type(question)]:<18}{mean(batched_values):>13.3f}"
        f"{mean(single_values):>12.3f}{stdev(batched_values):>13.4f}{stdev(single_values):>12.4f}"
    )

batched_cost = mean(cost for _values, cost, _latency in batched)
batched_latency = mean(latency for _values, _cost, latency in batched)
singles_cost = mean(
    sum(singles[run][key][1] for key in QUESTIONS) for run in range(RUNS)
)
singles_latency = mean(
    sum(singles[run][key][2] for key in QUESTIONS) for run in range(RUNS)
)
print(f"{'batching':<24}{'calls':>6}{'cost':>12}{'total time':>12}")
print(
    f"{f'one call, all {N}':<24}{1:>6}{'$' + format(batched_cost, '.6f'):>12}{format(batched_latency, '.2f') + 's':>12}"
)
print(
    f"{f'{N} calls, one each':<24}{N:>6}{'$' + format(singles_cost, '.6f'):>12}{format(singles_latency, '.2f') + 's':>12}"
)
print(
    f"\nbatching: {singles_cost / batched_cost:.1f}x cheaper, {singles_latency / batched_latency:.1f}x faster"
)

playground_link = make_playground_link({"article": DOCUMENT}, QUESTIONS)
display(
    Markdown(
        f"🔗 [Open this article + questions in the TypeSafe playground]({playground_link})"
    )
)
