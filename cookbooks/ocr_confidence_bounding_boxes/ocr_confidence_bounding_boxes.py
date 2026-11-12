# /// script
# requires-python = ">=3.10"
# dependencies = ["typesafe-sdk[pil]>=0.7.5", "pillow", "ipython", "cooksafe>=0.3.0,<0.4.0"]
# ///

"""OCR confidence and bounding boxes, from https://console.typesafe.ai/docs/cookbooks/ocr_confidence_bounding_boxes.

Needs no API key: it replays this cookbook's saved answers. To call the API,
set TYPESAFE_API_KEY and DOWNLOAD_CACHE = False.
"""

import io
import json
import os
import urllib.request
from pathlib import Path

from cooksafe import JsonCache
from IPython.display import display
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from typesafe_sdk import Box, Image, Noul, SystemOneResponse, TypeSafeClient

# Box questions need Jev 1.14 or later.
TYPESAFE_MODEL = "jev-1.14.0"  # the page's model; "jev-latest" runs the newest, live
BAR = 0.25  # a check scoring below this flags the field
ZOOM = 3  # the zoomed region is three times the first box, and at least ZOOM_MIN of the page
ZOOM_MIN = (0.30, 0.15)  # width, height as fractions of the page
GROW = (0.25, 0.20)  # review margin around each box, as fractions of its width and height
MIN_SIDE, MAX_SIDE = 280, 1568  # the pixel range Jev accepts for each side of an image

client = TypeSafeClient(
    api_key=os.environ.get("TYPESAFE_API_KEY", "cache-only"),  # keyless kernels replay the cache
    base_url=os.environ.get("TYPESAFE_ENDPOINT"),
    headers={"TypeSafe-Danger-Zone-Images": "true"},
    timeout=120.0,
)


EXAMPLES = (
    "https://raw.githubusercontent.com/typesafe-ai/typesafe-public-examples/main/"
    "cookbooks/ocr_confidence_bounding_boxes"
)


def download(name: str) -> Path:
    """Fetch one of this cookbook's files from typesafe-public-examples unless it is on disk."""
    path = Path(name)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{EXAMPLES}/{name}", path)
    return path


CACHE = Path("ocr_confidence_bounding_boxes_cache.json")
# True fills CACHE with the published answers, so the run makes no requests. To run
# live, set it to False and delete CACHE; a CACHE already on disk is never overwritten.
DOWNLOAD_CACHE = True
if DOWNLOAD_CACHE:
    download(CACHE.name)
json_cache = JsonCache(CACHE)

INPUTS = Path("inputs")  # the photos, keyed in extractions.json by file name
EXTRACTIONS = json.loads(download("extractions.json").read_text())
for photo in EXTRACTIONS:
    download(f"inputs/{photo}")
FIELDS = {
    # field: what it is, in the words every question uses
    "supplier_name": "the supplier's (seller's) business name",
    "invoice_number": "the invoice or document number",
    "invoice_date": "the invoice date",
    "amount_due": "the final total amount payable",
    "bank_name": "the name of the supplier's bank in the payment details",
    "account_holder": "the account holder name in the supplier's bank details",
    "account_number": "the supplier's bank account number",
    "branch_code": "the supplier's bank branch code",
}
SAME_VALUE = "Ignore differences in spacing, punctuation, currency symbols and letter case."

for photo, values in sorted(EXTRACTIONS.items()):
    with PILImage.open(INPUTS / photo) as page:
        print(f"{photo}  ({page.width} x {page.height} px)")
    for field, value in values.items():
        print(f"  {field:<16}{value}")

def as_image(picture: PILImage.Image, upscale_to: int | None = None) -> tuple[Image, tuple]:
    picture = picture.convert("RGB")
    if upscale_to and max(picture.size) < upscale_to:
        scale = upscale_to / max(picture.size)
        picture = picture.resize(
            (round(picture.width * scale), round(picture.height * scale)), PILImage.LANCZOS
        )
    picture.thumbnail((MAX_SIDE, MAX_SIDE))
    frame = (0.0, 0.0, 1.0, 1.0)  # left, top, width, height of the picture on the canvas
    if min(picture.size) < MIN_SIDE:
        canvas = PILImage.new("RGB", (max(picture.width, MIN_SIDE), max(picture.height, MIN_SIDE)), "white")
        x, y = (canvas.width - picture.width) // 2, (canvas.height - picture.height) // 2
        canvas.paste(picture, (x, y))
        frame = (x / canvas.width, y / canvas.height, picture.width / canvas.width,
                 picture.height / canvas.height)
        picture = canvas
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=90)
    return Image(buffer.getvalue()), frame


def crop(photo: str, box: dict, pad: float = 0.0) -> PILImage.Image:
    with PILImage.open(INPUTS / photo) as page:
        w, h = page.size
        return page.convert("RGB").crop((
            max(0, int((box["left"] - pad) * w)), max(0, int((box["top"] - pad) * h)),
            min(w, int((box["right"] + pad) * w)), min(h, int((box["bottom"] + pad) * h)),
        ))

def page_questions(values: dict) -> dict:
    questions = {}
    for field, value in sorted(values.items()):
        questions[f"box_{field}"] = Box(
            instructions=f"{FIELDS[field]}, printed as “{value}” in `page`"
        )
        questions[f"fits_{field}"] = Noul(
            instructions=f"Is “{value}” {FIELDS[field]} on `page`? {SAME_VALUE}"
        )
    return questions


def as_json(questions: dict) -> str:
    return json.dumps(
        {key: q.model_dump(mode="json", exclude_none=True) for key, q in questions.items()}
    )


@json_cache
def find_fields(photo: str, questions_json: str, model: str) -> dict:
    questions = json.loads(questions_json)
    fields = sorted(key.removeprefix("box_") for key in questions if key.startswith("box_"))
    with PILImage.open(INPUTS / photo) as page:
        image, _ = as_image(page)  # a whole page fills the size range, so no canvas
    response = client.system_one(
        state={"page": image}, questions=questions, model=model
    )
    return {
        "boxes": {f: response.boxes[f"box_{f}"].box.model_dump() for f in fields},
        "fits": {f: response.nouls[f"fits_{f}"].noul for f in fields},
    }


found = {photo: find_fields(photo, as_json(page_questions(values)), model=TYPESAFE_MODEL)
         for photo, values in sorted(EXTRACTIONS.items())}
for photo, result in found.items():
    print(photo)
    for field, value in EXTRACTIONS[photo].items():
        print(f"  {field:<16}{value:<24.24} value fits the page {result['fits'][field]:.2f}")

IMAGE = "{image}"  # stands for each question's own image until the request is sent
ZOOM_TASK = "Locate fields in zoomed-in parts of a supplier invoice."


def region_around(box: dict) -> dict:
    width = min(1.0, max(ZOOM_MIN[0], (box["right"] - box["left"]) * ZOOM))
    height = min(1.0, max(ZOOM_MIN[1], (box["bottom"] - box["top"]) * ZOOM))
    left = min(max(0.0, (box["left"] + box["right"] - width) / 2), 1 - width)
    top = min(max(0.0, (box["top"] + box["bottom"] - height) / 2), 1 - height)
    return {"left": left, "top": top, "right": left + width, "bottom": top + height}


def zoom_questions(values: dict) -> dict:
    return {
        field: Box(
            instructions=f"{FIELDS[field]}, printed as “{values[field]}” in this image: {IMAGE}"
        )
        for field in sorted(values)
    }


def ask_with_images(task: str, questions: dict, images: dict, model: str) -> SystemOneResponse:
    """Put each field's image into its question, and send all the questions in one request."""
    for field, question in questions.items():
        question["instructions"] = question["instructions"].replace(IMAGE, str(images[field]))
    with client.attach_assets(list(images.values())):  # resolves the references in the text
        return client.system_one(state={"task": task}, questions=questions, model=model)


@json_cache
def zoom_fields(
    photo: str, regions_json: str, task: str, questions_json: str, grow: tuple, model: str
) -> dict:
    regions, images, frames = json.loads(regions_json), {}, {}
    for field, region in regions.items():
        images[field], frames[field] = as_image(crop(photo, region), upscale_to=1200)
    response = ask_with_images(task, json.loads(questions_json), images, model)
    tight = {}
    for field, region in regions.items():
        b, (fx, fy, fw, fh) = response.boxes[field].box, frames[field]
        # canvas -> region -> page, then add the review margin
        left, right = (b.left - fx) / fw, (b.right - fx) / fw
        top, bottom = (b.top - fy) / fh, (b.bottom - fy) / fh
        gx, gy = (right - left) * grow[0], (bottom - top) * grow[1]
        rw, rh = region["right"] - region["left"], region["bottom"] - region["top"]
        tight[field] = {
            "left": max(0.0, region["left"] + (left - gx) * rw),
            "top": max(0.0, region["top"] + (top - gy) * rh),
            "right": min(1.0, region["left"] + (right + gx) * rw),
            "bottom": min(1.0, region["top"] + (bottom + gy) * rh),
        }
    return {"boxes": tight}

CROP_TASK = "Check values read from cropped parts of a supplier invoice."


def crop_questions(values: dict) -> dict:
    return {
        field: Noul(
            instructions=f"Image: {IMAGE}\nDoes this cropped image clearly show {FIELDS[field]}"
            f" as “{values[field]}”? {SAME_VALUE} Answer no if it is cut off, unreadable or"
            " different."
        )
        for field in sorted(values)
    }


@json_cache
def check_crops(
    photo: str, boxes_json: str, task: str, questions_json: str, model: str
) -> dict:
    images = {
        field: as_image(crop(photo, box, pad=0.01), upscale_to=800)[0]
        for field, box in json.loads(boxes_json).items()
    }
    response = ask_with_images(task, json.loads(questions_json), images, model)
    return {
        "shows": {f: response.nouls[f].noul for f in sorted(images)},
    }

def check_invoice(photo: str, values: dict) -> dict:
    first = find_fields(photo, as_json(page_questions(values)), model=TYPESAFE_MODEL)
    regions = {field: region_around(box) for field, box in first["boxes"].items()}
    tight = zoom_fields(photo, json.dumps(regions, sort_keys=True), ZOOM_TASK,
                        as_json(zoom_questions(values)), GROW, model=TYPESAFE_MODEL)
    shown = check_crops(photo, json.dumps(tight["boxes"], sort_keys=True), CROP_TASK,
                        as_json(crop_questions(values)), model=TYPESAFE_MODEL)
    return {
        field: {
            "value": values[field],
            "box": tight["boxes"][field],
            "fits": first["fits"][field],
            "shows": shown["shows"][field],
            "review": first["fits"][field] < BAR or shown["shows"][field] < BAR,
        }
        for field in sorted(values)
    }


checked = {photo: check_invoice(photo, values) for photo, values in sorted(EXTRACTIONS.items())}
KEY = {field: n for n, field in enumerate(FIELDS, 1)}  # the number drawn on each box below
print(f"{'photo':<15}{'#':<3}{'field':<16}{'extracted':<26}{'fits page':>10}{'crop':>7}  verdict")
for photo, fields in checked.items():
    for field, r in sorted(fields.items(), key=lambda item: KEY[item[0]]):
        verdict = "REVIEW" if r["review"] else "ok"
        print(f"{photo:<15}{KEY[field]:<3}{field:<16}{r['value']:<26.26}{r['fits']:>10.2f}"
              f"{r['shows']:>7.2f}  {verdict}")

BLUE, ORANGE = "#2a78d6", "#eb6834"


def annotate(photo: str, fields: dict) -> PILImage.Image:
    with PILImage.open(INPUTS / photo) as page:
        page = page.convert("RGB")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=max(14, page.width // 60))
    for field, r in fields.items():
        colour = ORANGE if r["review"] else BLUE
        b = r["box"]
        xy = (b["left"] * page.width, b["top"] * page.height,
              b["right"] * page.width, b["bottom"] * page.height)
        draw.rectangle(xy, outline=colour, width=max(3, page.width // 300))
        label = f" {KEY[field]} "  # a small badge keeps fields on one printed line legible
        tx, ty = xy[0], max(0, xy[1] - font.size - 4)
        draw.rectangle(draw.textbbox((tx, ty), label, font=font), fill=colour)
        draw.text((tx, ty), label, fill="white", font=font)
    return page


for photo, fields in checked.items():
    display(annotate(photo, fields))
