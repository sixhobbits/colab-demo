# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.0", "ipython", "cooksafe>=0.3.0,<0.4.0"]
# ///

"""Simple memory system, from https://console.typesafe.ai/docs/cookbooks/basic_memory.

Needs no API key: it replays this cookbook's saved answers. To call the API,
set TYPESAFE_API_KEY and DOWNLOAD_CACHE = False.
"""

import json
import os
import urllib.request
from pathlib import Path

from cooksafe import JsonCache, make_playground_link
from IPython.display import Markdown, display
from typesafe_sdk import Choice, Noul, NoulCriteria, SystemOneResponse, TypeSafeClient

TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
REMEMBER_AT = 0.5  # save when P(lasting customer fact) reaches this starting threshold
MEMORY_KEY = "earlier_tickets_from_this_customer"  # where saved tickets go in the state

client = TypeSafeClient(
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),  # keyless kernels replay the cache
    base_url=os.environ.get("TYPESAFE_ENDPOINT"),
    timeout=120.0,
)

EXAMPLES_URL = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/basic_memory"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        urllib.request.urlretrieve(f"{EXAMPLES_URL}/{name}", path)
    return path


CACHE = Path("basic_memory_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)

DATA = {
    "customers": {
        "C-1042": {"id": "C-1042", "company": "Acme Corp"},
        "C-2210": {"id": "C-2210", "company": "Example Studio"},
    },
    "earlier_tickets": [
        {"id": "T-101", "customer_id": "C-1042", "received_on": "2026-08-04",
         "subject": "Adding an admin",
         "text": "Hi, we run your self-hosted edition on Windows Server 2022 in our own data "
                 "centre. How do I add a new admin user?"},
        {"id": "T-102", "customer_id": "C-2210", "received_on": "2026-08-11",
         "subject": "Exporting to PDF",
         "text": "We're a small design studio on your cloud edition, paying monthly on the "
                 "self-serve Team plan. How do I export a dashboard as a PDF?"},
        {"id": "T-103", "customer_id": "C-1042", "received_on": "2026-08-19",
         "subject": "Invoice download",
         "text": "Sorry for the typos, typing this on my phone at the airport. Where can I "
                 "download last month's invoice?"},
        {"id": "T-104", "customer_id": "C-2210", "received_on": "2026-09-02",
         "subject": "Dark mode?",
         "text": "Our whole team is out on Friday for a public holiday, so no rush on this "
                 "one. Is there a dark mode?"},
        {"id": "T-105", "customer_id": "C-1042", "received_on": "2026-09-10",
         "subject": "Sandbox environment",
         "text": "For context, we signed an Enterprise contract with you in March. Can we get "
                 "a sandbox environment for testing?"},
    ],
    "later_messages": [
        {"subject": "Slow dashboards",
         "text": "Dashboards have been really slow since this morning."},
        {"subject": "Upgrade failing",
         "text": "The upgrade to version 5.2 keeps failing halfway through."},
        {"subject": "More seats",
         "text": "We'd like to add 20 more seats before next quarter."},
    ],
    "later_received_on": "2026-09-28",
    # Labelled by hand from what each customer said in the earlier tickets.
    "expected_team": {
        "C-1042": {"Slow dashboards": "selfhosted_windows",
                   "Upgrade failing": "selfhosted_windows", "More seats": "enterprise_desk"},
        "C-2210": {"Slow dashboards": "cloud_ops", "Upgrade failing": "cloud_ops",
                   "More seats": "billing"},
    },
}
CUSTOMERS = DATA["customers"]


def ticket_state(customer_id: str, ticket_id: str, received_on: str, subject: str,
                 text: str) -> dict:
    """The state Jev sees for one ticket: a simplified ticket, like one from a helpdesk."""
    return {
        "customer": CUSTOMERS[customer_id],
        "ticket": {
            "id": ticket_id,
            "received_on": received_on,
            "subject": subject,
            "messages": [{"from": "customer", "text": text}],
        },
    }


EARLIER = [
    ticket_state(t["customer_id"], t["id"], t["received_on"], t["subject"], t["text"])
    for t in DATA["earlier_tickets"]
]
print(f"{len(CUSTOMERS)} customers, {len(EARLIER)} earlier tickets, "
      f"{len(DATA['later_messages'])} later messages. The first ticket as Jev sees it:\n")
print(json.dumps(EARLIER[0], indent=2))

ROUTE = {
    "team": Choice(
        instructions="Which support team should handle `ticket`?",
        criteria={
            "cloud_ops": "Cloud team: problems with the hosted cloud edition, run by us",
            "selfhosted_linux": "Self-hosted team, Linux: installs, upgrades and "
            "performance of the self-hosted edition on Linux servers",
            "selfhosted_windows": "Self-hosted team, Windows: installs, upgrades and "
            "performance of the self-hosted edition on Windows Server",
            "billing": "Billing: invoices, payments, seats and plan changes for "
            "self-serve plans",
            "enterprise_desk": "Enterprise desk: seats, contracts and account changes for "
            "customers on an Enterprise contract (their technical problems go to the "
            "technical teams)",
            "general": "General product questions: features, how-to, settings",
            "none_suitable": "None of these teams fits the ticket",
        },
    ),
}


@json_cache
def _system_one(state: str, questions: str, model: str) -> dict:
    response = client.system_one(
        state=json.loads(state), questions=json.loads(questions), model=model
    )
    return response.model_dump(mode="json", exclude={"model"})


def ask(state: dict, questions: dict) -> SystemOneResponse:
    """client.system_one, with the model pinned and the call cached."""
    as_json = {qid: q.model_dump(mode="json", exclude_none=True) for qid, q in questions.items()}
    raw = _system_one(json.dumps(state), json.dumps(as_json), model=TYPESAFE_MODEL)
    # The cached answer stores no model; the response type needs one, so use the one it is keyed on.
    return SystemOneResponse.model_validate({**raw, "model": TYPESAFE_MODEL})


def later_state(customer_id: str, message_index: int) -> dict:
    """One of the later messages, sent by this customer, with a fixed ticket ID."""
    n = 201 + 3 * list(CUSTOMERS).index(customer_id) + message_index
    message = DATA["later_messages"][message_index]
    return ticket_state(customer_id, f"T-{n}", DATA["later_received_on"], message["subject"],
                        message["text"])


slow = later_state("C-1042", 0)
team = ask(slow, ROUTE).answers["team"]
print(f"C-1042: {slow['ticket']['messages'][0]['text']!r}")
print(f"  plain Jev -> {team.choice} {team.probabilities[team.choice]:.2f}")

REMEMBER = {
    "remember": Noul(
        instructions=(
            "In `request.ticket`, does the thread state a lasting fact about the customer that "
            "should change how their future, unrelated tickets are handled? Count what the "
            "customer or our support staff wrote in `request.ticket`. `jev_answered` shows how "
            "the ticket was routed, which may rely on things learned earlier: do not count "
            "anything it says or implies."
        ),
        criteria=NoulCriteria(
            true=(
                "Yes: in `request.ticket` the customer or our support staff state something "
                "about the customer that stays true after this ticket, such as the product "
                "edition or operating system they run, their plan or contract, or a standing "
                "preference."
            ),
            false=(
                "No: `request.ticket` only contains things specific to this one ticket or "
                "moment, such as a one-off request, today's schedule, or the device they "
                "happen to be typing on; or the lasting fact appears only in `jev_answered`."
            ),
        ),
    ),
}


def answer_text(question, answer):
    """The readable part of an answer: the chosen option's description, or the number."""
    if answer.type == "choice":
        return question.criteria[answer.choice]
    return answer.noul if answer.type == "noul" else answer.score


def as_memory(state: dict, questions: dict, response: SystemOneResponse) -> dict:
    return {
        "request": state,
        "jev_answered": {
            q.instructions: answer_text(q, response.answers[qid])
            for qid, q in questions.items()
        },
    }


def remember_probability(memory: dict) -> float:
    """P(the request states an important fact about the customer), with the answer as context."""
    return ask(memory, REMEMBER).answers["remember"].noul

MEMORIES: dict[str, list[dict]] = {}  # customer ID -> that customer's saved memories


def customer_id(state: dict) -> str:
    return state["customer"]["id"]


def ask_with_memory(state: dict, questions: dict) -> SystemOneResponse:
    """ask(), plus memory: same arguments, same return value."""
    # 1. Answer the call with this customer's existing memories.
    saved = MEMORIES.get(customer_id(state), [])
    response = ask({**state, MEMORY_KEY: saved} if saved else state, questions)

    # 2. Ask Jev whether the original request states an important fact worth keeping.
    memory = as_memory(state, questions, response)
    if remember_probability(memory) >= REMEMBER_AT:
        MEMORIES.setdefault(customer_id(state), []).append(memory)
    return response

PROBES = {"Slow dashboards": 0, "More seats": 2}  # later messages to send after each ticket


def memory_ids(cid: str) -> str:
    return ", ".join(m["request"]["ticket"]["id"] for m in MEMORIES.get(cid, [])) or "none"


def send_probes(cid: str) -> None:
    for subject, i in PROBES.items():
        team = ask_with_memory(later_state(cid, i), ROUTE).answers["team"]
        print(f"      {subject:<16} -> {team.choice:<19} {team.probabilities[team.choice]:.2f}")


for cid in CUSTOMERS:
    print(f"{cid}, memories: {memory_ids(cid)}")
    send_probes(cid)

for state in EARLIER:
    cid = customer_id(state)
    response = ask_with_memory(state, ROUTE)
    p = remember_probability(as_memory(state, ROUTE, response))  # cached: the wrapper's call
    verdict = "saved" if p >= REMEMBER_AT else "not saved"
    text = state["ticket"]["messages"][0]["text"]
    print(f"\n{cid} {state['ticket']['id']}  {response.answers['team'].choice:<19} "
          f"P(remember) {p:.2f}  {verdict}")
    print(f"    {text[:84]}")
    print(f"{cid}, memories: {memory_ids(cid)}")
    send_probes(cid)

EXPECTED = DATA["expected_team"]
plain_correct = memory_correct = total = 0
example_state = None


def cell(answer, expected: str) -> str:
    mark = "✓" if answer.choice == expected else "✗"
    return f"{mark} {answer.choice} {answer.probabilities[answer.choice]:.2f}"


print(f"{'customer':<9}{'message':<17}{'expected':<20}{'plain Jev':<27}{'with memory':<27}")
for cid in CUSTOMERS:
    for i, message in enumerate(DATA["later_messages"]):
        state = later_state(cid, i)
        expected = EXPECTED[cid][message["subject"]]
        plain = ask(state, ROUTE).answers["team"]
        if example_state is None:
            # Keep the exact state used for this call for the playground link.
            saved = MEMORIES.get(cid, [])
            example_state = json.loads(json.dumps({**state, MEMORY_KEY: saved} if saved else state))
        response = ask_with_memory(state, ROUTE)
        remembered = response.answers["team"]
        plain_correct += plain.choice == expected
        memory_correct += remembered.choice == expected
        total += 1
        print(f"{cid:<9}{message['subject']:<17}{expected:<20}{cell(plain, expected):<27}"
              f"{cell(remembered, expected):<27}")

print(f"\nExpected routes: {plain_correct}/{total} without memory, "
      f"{memory_correct}/{total} with memory.")

playground_link = make_playground_link(example_state, ROUTE)
display(Markdown(
    f"🔗 [Open C-1042's ticket with its memories in the TypeSafe playground]({playground_link})"
))
