# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk>=0.7.5,<0.8.0", "pillow", "matplotlib", "ipython", "cooksafe>=0.3.0,<0.4.0"]
# ///

"""Feeder-cam visit log, from https://console.typesafe.ai/docs/cookbooks/feeder_watch.

Needs no API key: it replays this cookbook's saved answers. To call the API,
set TYPESAFE_API_KEY and DOWNLOAD_CACHE = False.
"""

import json
import os
import statistics
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from cooksafe import JsonCache
from typesafe_sdk import Box, Choice, Image, Noul, SystemOneResponse, TypeSafeClient

# Box questions need Jev 1.14 or later.
TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}  # images and Box are opt-in

client = TypeSafeClient(
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),  # keyless runs replay the cache
    timeout=60.0,
)

EXAMPLES = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/feeder_watch"
)
FRAMES = Path("usfws_frames")
LABELS = Path("usfws_labels.json")


def download(url: str, path: Path) -> Path:
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path

CACHE = Path("feeder_watch_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(f"{EXAMPLES}/{CACHE.name}", CACHE)
json_cache = JsonCache(CACHE)


if not FRAMES.exists():
    with zipfile.ZipFile(download(f"{EXAMPLES}/usfws_frames.zip", Path("usfws_frames.zip"))) as zf:
        zf.extractall()  # the zip holds usfws_frames/u001.jpg to u101.jpg
NAMES = [f"u{second:03d}.jpg" for second in range(1, 102)]  # one frame per second
paths = [FRAMES / n for n in NAMES]
labels = json.loads(download(f"{EXAMPLES}/usfws_labels.json", LABELS).read_text())

birds_per_frame = [len(labels[n]["species"]) for n in NAMES]
print(f"{len(paths)} frames, {PILImage.open(paths[0]).size[0]} x {PILImage.open(paths[0]).size[1]}")
print(f"frames by birds in full view: {[birds_per_frame.count(k) for k in (0, 1, 2)]} (0, 1, 2)")
print("u010.jpg is labeled", labels["u010.jpg"])

SPECIES = {
    "blue_jay": "A blue bird with a crest, a pale grey-white face and chest, a black necklace, "
    "and black bars on the blue wings and tail.",
    "cardinal_male": "A bright red bird with a pointed crest, a black mask, and an orange beak.",
    "cardinal_female": "A warm brown or buff bird with a pointed crest, reddish tinges on the "
    "wings, tail and crest, and an orange beak.",
    "grackle": "A common grackle: a glossy black bird with a pale yellow eye, a long heavy dark "
    "beak, and an iridescent blue, purple or bronze sheen on the head and body. On a close-up "
    "feeder camera it often fills the frame, so only the glossy dark head, pale eye and beak "
    "may be visible.",
    "starling": "A European starling: a chunky short-tailed dark bird with glossy plumage "
    "speckled with pale spots, a dark eye, and a pointed yellow or dark beak. Juveniles are "
    "plain grey-brown with a dark beak.",
    "mourning_dove": "A plump grey-brown dove with a small head, a long pointed tail, and black "
    "spots on the wings.",
    "house_sparrow": "A small streaky brown sparrow. Males have a grey crown and a black bib.",
    "house_finch": "A small streaky brown finch. Males have a rosy red head and chest.",
    "red_winged_blackbird": "A glossy black bird with a red shoulder patch edged in yellow.",
    "downy_woodpecker": "A small black-and-white woodpecker with a white stripe down the back "
    "and spotted wings. Males have a red patch on the back of the head.",
    "other": "A bird that matches none of the other descriptions.",
}

QUESTIONS = {
    "bird_present": Noul(instructions="Is there at least one bird in `frame`?"),
    "how_many": Choice(
        instructions="How many birds are in `frame`?",
        criteria={
            "none": "No bird at all.",
            "one": "Exactly one bird.",
            "several": "Two or more birds.",
        },
    ),
    "bird": Box(instructions="The most prominent bird in `frame`"),
    "species_left": Choice(
        instructions="What species is the leftmost bird in `frame`?", criteria=SPECIES
    ),
    "species_right": Choice(
        instructions="What species is the rightmost bird in `frame`?", criteria=SPECIES
    ),
}

@json_cache
def ask_cached(frame: str, questions_json: str, model: str) -> dict:
    started = perf_counter()
    response = client.system_one(
        state={"frame": Image(FRAMES / frame)},
        questions=json.loads(questions_json),
        model=model,
        extra_headers=IMAGE_HEADERS,
    )
    seconds = round(perf_counter() - started, 2)
    return {"response": response.model_dump(mode="json", exclude={"model"}), "seconds": seconds}


def ask(path: Path, questions: dict):
    as_json = json.dumps(
        {key: q.model_dump(mode="json", exclude_none=True) for key, q in questions.items()},
        sort_keys=True,
    )
    cached = ask_cached(path.name, as_json, model=TYPESAFE_MODEL)
    # The cached answer stores no model; the response type needs one, so use the one it is keyed on.
    response = SystemOneResponse.model_validate({**cached["response"], "model": TYPESAFE_MODEL})
    return response, cached["seconds"]


def edges(box) -> str:
    return f"left {box.left:.2f}  right {box.right:.2f}  top {box.top:.2f}  bottom {box.bottom:.2f}"


def show_first_request(name: str) -> None:
    response, seconds = ask(FRAMES / name, QUESTIONS)
    how_many = response.choices["how_many"]
    print(f"{name}  labeled {labels[name]['species'] or 'empty'}  ({seconds:.2f} s)")
    print(f"  bird_present   {response.nouls['bird_present'].noul:.2f}")
    print(f"  how_many       {how_many.choice} {how_many.probabilities[how_many.choice]:.2f}")
    bird = response.boxes["bird"]
    print(f"  bird           {edges(bird.box)}  confidence {bird.confidence:.2f}")
    for key in ("species_left", "species_right"):
        answer = response.choices[key]
        print(f"  {key:<15}{answer.choice} {answer.probabilities[answer.choice]:.2f}")
    print()


for name in ("u010.jpg", "u090.jpg", "u097.jpg"):
    show_first_request(name)

POSITION_BOXES = {
    "leftmost": Box(instructions="The leftmost bird in `frame`"),
    "rightmost": Box(instructions="The rightmost bird in `frame`"),
}
by_position, _ = ask(FRAMES / "u010.jpg", POSITION_BOXES)
for key in POSITION_BOXES:
    answer = by_position.boxes[key]
    print(f"{key:<11}{edges(answer.box)}  confidence {answer.confidence:.2f}")

BOX_TARGETS = {
    "blue_jay": "the blue jay",
    "cardinal_male": "the male northern cardinal (the bright red bird)",
    "cardinal_female": "the female northern cardinal (the brown bird with an orange beak)",
    "grackle": "the common grackle (the glossy black bird)",
    "starling": "the European starling (the speckled dark bird)",
    "mourning_dove": "the mourning dove",
    "house_sparrow": "the house sparrow",
    "house_finch": "the house finch",
    "red_winged_blackbird": "the red-winged blackbird",
    "downy_woodpecker": "the downy woodpecker (the black and white bird)",
    "other": "the bird that is none of the listed species",
}


def box_questions(species: list[str]) -> dict:
    return {sp: Box(instructions=f"{BOX_TARGETS[sp]} in `frame`") for sp in species}


by_species, _ = ask(FRAMES / "u010.jpg", box_questions(["cardinal_male", "downy_woodpecker"]))
for key, answer in by_species.boxes.items():
    print(f"{key:<17}{edges(answer.box)}  confidence {answer.confidence:.2f}")

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE = "#2a78d6", "#eb6834"
COLOURS = {"cardinal_male": ORANGE, "downy_woodpecker": BLUE}  # every other species: INK2
FONT = ImageFont.load_default(size=26)


def draw(ax, name: str, boxes: list[tuple[str, dict, str]], title: str) -> None:
    """Draw (label, edges, colour) boxes on a frame and put it on a matplotlib axis."""
    image = PILImage.open(FRAMES / name).convert("RGB")
    width, height = image.size
    pen = ImageDraw.Draw(image)
    for label, e, colour in boxes:
        x0, y0 = e["left"] * width, e["top"] * height
        pen.rectangle([x0, y0, e["right"] * width, e["bottom"] * height], outline=colour, width=6)
        text_box = pen.textbbox((x0, y0), label, font=FONT)
        y = y0 - (text_box[3] - text_box[1]) - 12 if y0 > 40 else y0 + 6
        pen.rectangle([x0, y, x0 + text_box[2] - text_box[0] + 12, y + 36], fill=colour)
        pen.text((x0 + 6, y + 2), label, fill="white", font=FONT)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=10, color=INK2)


def show(fig, filename: str) -> None:
    """Show a figure in a notebook; save it as a PNG when run as a script."""
    if __name__ == "__main__" and "__file__" in globals():
        fig.savefig(filename, dpi=120)
        print(f"saved {filename}")
    else:
        display(fig)
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
draw(
    axes[0],
    "u010.jpg",
    [(key, by_position.boxes[key].box.model_dump(), MUTED) for key in POSITION_BOXES],
    "Box by position: both boxes on the cardinal",
)
draw(
    axes[1],
    "u010.jpg",
    [(key.replace("_", " "), a.box.model_dump(), COLOURS[key]) for key, a in by_species.boxes.items()],
    "Box by species name: one box per bird",
)
fig.suptitle("The second request on frame u010 (0:09)", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
show(fig, "second-request.png")

OVERLAP_AT = 0.6  # two named boxes overlapping this much are one bird found twice


def overlap(a: dict, b: dict) -> float:
    """Intersection over union of two boxes given as edge fractions."""
    w = max(0.0, min(a["right"], b["right"]) - max(a["left"], b["left"]))
    h = max(0.0, min(a["bottom"], b["bottom"]) - max(a["top"], b["top"]))
    area = lambda e: (e["right"] - e["left"]) * (e["bottom"] - e["top"])  # noqa: E731
    union = area(a) + area(b) - w * h
    return w * h / union if union > 0 else 0.0


def dedupe(birds: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for bird in sorted(birds, key=lambda b: -b["box_confidence"]):
        if all(overlap(bird["box"], k["box"]) < OVERLAP_AT for k in kept):
            kept.append(bird)
    return kept

PRESENT_AT = 0.5  # below this, the frame is empty and the other answers are ignored
SPECIES_AT = 0.5  # a species answer below this is too unsure to name a box


def ask_frame(path: Path) -> dict:
    first, seconds = ask(path, QUESTIONS)
    result = {
        "frame": path.name,
        "bird_present": first.nouls["bird_present"].noul,
        "how_many": first.choices["how_many"].choice,
        "birds": [],
        "requests": 1,
        "seconds": seconds,
    }
    # the species named by the leftmost and rightmost answers, each with its best probability
    named: dict[str, float] = {}
    sure_sides = 0
    for key in ("species_left", "species_right"):
        answer = first.choices[key]
        p = answer.probabilities[answer.choice]
        if p >= SPECIES_AT:
            named[answer.choice] = max(p, named.get(answer.choice, 0.0))
            sure_sides += 1
    if result["bird_present"] < PRESENT_AT or not named:
        return result

    # with several birds, the `bird` box only fits when both sides name the same species
    if result["how_many"] == "several" and (len(named) == 2 or sure_sides < 2):
        second, more_seconds = ask(path, box_questions(list(named)))
        result["requests"] = 2
        result["seconds"] += more_seconds
        boxes = {sp: second.boxes[sp] for sp in named}
    else:
        top = max(named, key=named.get)
        named, boxes = {top: named[top]}, {top: first.boxes["bird"]}

    birds = [
        {"species": sp, "p": p, "box": boxes[sp].box.model_dump(), "box_confidence": boxes[sp].confidence}
        for sp, p in named.items()
    ]
    result["birds"] = dedupe(birds)
    return result

with ThreadPoolExecutor(max_workers=4) as pool:  # 101 frames, about 180 requests, 4 at a time
    results = list(pool.map(ask_frame, paths))

two_requests = [r for r in results if r["requests"] == 2]
print(f"{len(results)} frames, {sum(r['requests'] for r in results)} requests")
print(f"frames that needed the second request: {len(two_requests)}")
print(f"frames that ended with two boxes:      {sum(len(r['birds']) == 2 for r in results)}")
print(f"seconds per frame, median: {statistics.median(r['seconds'] for r in results):.2f}")

def visits(results: list[dict], species_at=0.6, max_gap=2, min_frames=2) -> list[dict]:
    done, open_visits, last_seen = [], {}, {}
    for second, result in enumerate(results):  # one frame per second
        seen = {b["species"] for b in result["birds"] if b["p"] >= species_at}
        for sp in list(open_visits):
            if sp not in seen and second - last_seen[sp] > max_gap:
                done.append(open_visits.pop(sp))
        for sp in sorted(seen):
            if sp in open_visits:
                open_visits[sp]["end"] = second
                open_visits[sp]["frames"] += 1
            else:
                open_visits[sp] = {"species": sp, "start": second, "end": second, "frames": 1}
            last_seen[sp] = second
    done.extend(open_visits.values())
    kept = [v for v in done if v["frames"] >= min_frames]
    return sorted(kept, key=lambda v: (v["start"], v["species"]))


def clock(second: int) -> str:
    return f"{second // 60}:{second % 60:02d}"


def print_visits(found: list[dict]) -> None:
    for v in found:
        seconds = v["end"] - v["start"] + 1
        print(f"  {v['species']:<18}{clock(v['start'])} to {clock(v['end'])}  ({seconds} s)")


found = visits(results)
print("Visits:")
print_visits(found)

per_species: dict[str, list[int]] = {}
for v in found:
    count = per_species.setdefault(v["species"], [0, 0])
    count[0] += 1
    count[1] += v["end"] - v["start"] + 1
print("\nPer species:")
for sp, (n, seconds) in sorted(per_species.items(), key=lambda kv: -kv[1][1]):
    print(f"  {sp:<18}{n} visit{'s' if n > 1 else ''}, {seconds} s at the feeder")
print("\nVisits in the hand labels:")
print_visits(visits([{"birds": [{"species": s, "p": 1.0} for s in labels[n]["species"]]} for n in NAMES]))

def draw_visits(found: list[dict], seconds: int, title: str) -> None:
    by_species: dict[str, list[dict]] = {}
    for v in found:
        by_species.setdefault(v["species"], []).append(v)
    total = {sp: sum(v["end"] - v["start"] + 1 for v in vs) for sp, vs in by_species.items()}
    order = sorted(by_species, key=lambda sp: -total[sp])

    height = 0.45 * len(order) + 1.4
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11, height), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [3.6, 1], "wspace": 0.04},
    )
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.spines[["top", "right", "left"]].set_visible(False)
        a.tick_params(colors=INK2, labelsize=9, length=0)
    for row, sp in enumerate(order):
        colour = MUTED if sp == "other" else BLUE
        spans = [(v["start"], v["end"] - v["start"] + 1) for v in by_species[sp]]
        ax.broken_barh(spans, (row - 0.32, 0.64), facecolors=colour)
        bx.barh(row, total[sp], height=0.64, color=colour)
        n = len(by_species[sp])
        label = f"{total[sp]} s, {n} visit{'s' if n > 1 else ''}"
        bx.text(total[sp] + seconds * 0.03, row, label, va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(order)), [sp.replace("_", " ") for sp in order], color=INK, fontsize=10)
    ax.set_xlim(0, seconds)
    ax.set_xticks(range(0, seconds + 1, 10), [clock(s) for s in range(0, seconds + 1, 10)])
    ax.grid(axis="x", color="#e4e3df")
    ax.set_axisbelow(True)
    ax.set_title("Each visit, by time of footage (m:ss)", loc="left", fontsize=10, color=INK2)
    bx.set_xlim(0, seconds * 1.6)
    bx.set_xticks([])
    bx.set_yticks([])
    bx.spines["bottom"].set_visible(False)
    bx.set_title("Time at the feeder", loc="left", fontsize=10, color=INK2)
    for a in (ax, bx):
        a.set_ylim(len(order) - 0.5, -0.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    # margins in inches: the species names on the left, the titles on top, the times below
    fig.subplots_adjust(left=1.6 / 11, right=0.99, top=1 - 0.75 / height, bottom=0.4 / height)
    show(fig, "visits.png")


draw_visits(found, len(results), f"{len(found)} visits in {clock(len(results))} of footage")

logged = {(v["species"], v["start"]) for v in visits(results)}
print("Single-frame sightings that min_frames=2 leaves out:")
print_visits([v for v in visits(results, min_frames=1) if (v["species"], v["start"]) not in logged])

present_ok, found, total, misses = 0, 0, 0, []
for r in results:
    label = labels[r["frame"]]
    has_bird = bool(label["species"]) or label["partial"]  # a partial bird is still a bird
    if (r["bird_present"] >= PRESENT_AT) == has_bird:
        present_ok += 1
    else:
        misses.append(f"{r['frame']}  bird_present {r['bird_present']:.2f}, labeled {label['species'] or 'empty'}")
    if label["partial"]:
        continue
    returned = {b["species"] for b in r["birds"]}
    for sp in label["species"]:
        total += 1
        if sp in returned:
            found += 1
        else:
            misses.append(f"{r['frame']}  labeled {sp}, returned {sorted(returned) or 'nothing'}")

print(f"bird_present agrees with the labels: {present_ok}/{len(results)} frames")
print(f"labeled birds returned:              {found}/{total}")
print("\nDisagreements:")
print("\n".join(misses) or "none")

by_frame = {r["frame"]: r for r in results}


def answer_boxes(name: str) -> list[tuple[str, dict, str]]:
    return [
        (f"{b['species'].replace('_', ' ')} {b['p']:.2f}", b["box"], COLOURS.get(b["species"], INK2))
        for b in by_frame[name]["birds"]
    ]


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
for ax, name in zip(axes, ("u094.jpg", "u076.jpg")):
    r = by_frame[name]
    second = int(name[1:4]) - 1
    draw(ax, name, answer_boxes(name), f"{clock(second)}  bird_present {r['bird_present']:.2f}")
fig.suptitle("Two hard frames from the run", x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout()
show(fig, "failures.png")
