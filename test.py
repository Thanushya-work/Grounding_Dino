# # # from pydub import AudioSegment

# # # final = AudioSegment.empty()

# # # for i in range(1, 10):
# # #     final += AudioSegment.from_mp3(f"{i}.mp3")

# # # final.export("final_quiz_intro.mp3", format="mp3")

# # {1 : [51,52,53,54,55,56,57,58,59,60]}

# # import easyocr
# # import cv2
# # import re

# # image_path = "img.png"   

# # # Load image
# # image = cv2.imread(image_path)

# # # Initialize EasyOCR reader
# # reader = easyocr.Reader(['en'])

# # # Perform OCR
# # results = reader.readtext(image)

# # # Extract text
# # detected_texts = [res[1] for res in results]

# # print("Detected Text:", detected_texts)

# # # Find MFD date pattern (dd/mm/yy or dd/mm/yyyy)
# # date_pattern = r'\b\d{2}/\d{2}/\d{2,4}\b'

# # mfd_date = None

# # for text in detected_texts:
# #     match = re.search(date_pattern, text)
# #     if match:
# #         mfd_date = match.group()
# #         break

# # if mfd_date:
# #     print("Extracted MFD Date:", mfd_date)
# # else:
# #     print("MFD Date not found")


# import base64
# from openai import OpenAI

# # 🔑 Initialize client
# client = OpenAI(api_key="")

# # 📸 Encode local image
# def encode_image(image_path):
#     with open(image_path, "rb") as img:
#         return base64.b64encode(img.read()).decode("utf-8")

# # 👉 Change this to your image
# image_path = "IMG-MO1FH46F-A27O.jpg"

# base64_image = encode_image(image_path)

# # 🔥 EXACT FINAL PROMPT (same as I gave you)
# PROMPT = """
# You are an expert AI system trained to detect Coca-Cola retail activation elements from store images with high precision. You have been trained on reference visuals for each element class.

# ## Task
# Evaluate ALL 19 Coca-Cola branded activation element classes listed below against the provided image. For every class, you MUST return a result — either "Y" (present) or "N" (not present). Do NOT skip any class.

# ---

# ## Core Detection Philosophy

# - **Detect anything with visible Coca-Cola Company branding — this includes the parent brand AND all portfolio sub-brands listed below. An element showing ANY of these brands qualifies for detection:
# BrandKey Visual IdentifiersCoca-Cola / CokeRed background, white Spencerian script logo, contour bottleSpriteGreen + yellow color scheme, "Sprite" wordmark, lemon-lime imageryFantaOrange (or red/grape/green per flavor), bubbly "Fanta" wordmarkKinleyLight blue/white, "Kinley" wordmark, water/soda brandingThums UpDark red/black, thunder bolt logo, "Thums Up" wordmarkMaazaYellow-orange, mango imagery, "Maaza" wordmarkLimcaLight blue-green, lemon imagery, "Limca" wordmarkMinute MaidOrange/white, juice imagery, "Minute Maid" wordmarkSchweppesDark blue/gold, "Schweppes" wordmark
# Important: Sub-brand-only elements (e.g., a pure Sprite poster with no Coca-Cola logo) are VALID detections. You do NOT need the Coca-Cola master logo to be present — sub-brand identity alone is sufficient.
# - **Partial visibility is acceptable**: If the element is partially cut off or at image edges but Coca-Cola branding is still recognizable, attempt classification. You do NOT need 40% visibility — if the branding is identifiable, try to classify it.
# - **Do NOT guess**: If branding is genuinely ambiguous, blurry, or obscured, mark as "N".
# - **One detection per physical object**: Do NOT assign two classes to the same object. Use Similarity Resolution Logic below to pick the best one.
# - **Never ignore Coca-Cola products themselves**: Bottles, cans, or branded product units can help confirm element context (e.g., a rack holding Coke bottles is an MT_RACK_DISPLAY).

# ---

# ## Detection Classes (with Visual Reference Descriptions)

# ### POSTER (Class ID: 1019)
# Flat printed material fixed to a wall, pillar surface, or glass — NOT hanging or freestanding. Typically portrait or landscape rectangular. May show campaign artwork, price, or brand message. Usually A2–A0 size range.

# ### STREAMER (Class ID: 1020)
# Long, thin horizontal or vertical printed strip — typically hung across a wall, ceiling, or stretched between two points. Series of repeated brand panels in a strip format. Often seen in multiples running across the top of a shop.

# ### DANGLER (Class ID: 1081)
# A printed panel hanging from the ceiling or overhead structure via string/wire. Typically double-sided, suspended freely in space. Smaller than a banner. Moves/sways slightly when hanging.

# ### WOBBLER (Class ID: 1082)
# A small, flexible printed tag physically attached to a shelf edge. Sticks out from the shelf, often teardrop or rounded rectangular shaped. Used for price callouts or product highlights. Usually 10–20 cm wide.

# ### WALL_BRANDING (Class ID: 1022)
# Large-format Coca-Cola branding covering a significant portion of a shop wall — painted mural, large vinyl, or printed sheet. Typically covers an entire wall or large section. May include brand ambassador imagery.

# ### TABLE STICKER (CLASS ID: 1027)

# Detect Coca-Cola branded stickers or printed branding applied directly on tables (dining tables, customer seating tables). These are flat graphics placed on the tabletop surface and often include Coca-Cola logo, campaign visuals, or food pairing imagery.


# ### DPS (DIGITAL POSTER SIGNAGE) (CLASS ID: 1053)

# Detect any large shop signboard or name board visible on the exterior of the store that contains Coca-Cola or its sub-brand branding. This includes boards placed above the दुकान/entrance showing the shop name along with Coca-Cola branding (e.g., Coca-Cola logo, bottle visuals, brand colors like red/white). The board is typically fixed, non-digital, and forms part of the shop’s identity signage.

# ### RGB CRATE STACKING (CLASS ID: 1056)

# Detect stacks of Coca-Cola branded plastic crates (typically red color) used for storing glass bottles. These crates are usually stacked on top of each other either inside or outside the shop and are used for inventory/storage purposes.

# ### FRONT_WINDOW_BRANDING (Class ID: 1083)
# Branded sticker, vinyl, or print applied directly onto the shop's glass window or door. Viewed from outside the store. May be partially transparent. Common at shop entrance.

# ### CEILING_PILLAR_BRANDING (Class ID: 1084)
# Branding that wraps or covers a vertical pillar/column that extends toward or meets the ceiling. Full or partial wrap. Often seen in modern retail environments.

# ### PILLAR_BRANDING (Class ID: 1040)
# Branding applied to a standalone vertical pillar or column — not necessarily reaching the ceiling. May be at shop entrance, outside the store, or standalone column branding on the exterior.

# ### COUNTER_FRONT (Class ID: 1085)
# Coca-Cola branded material applied on the vertical front face of a billing/checkout counter. The branding faces customers standing in front of the counter. May be a printed sheet, sticker, or vinyl wrap.

# ### SHELF_RACK_BRANDING (Class ID: 1086)
# Branding strips, printed panels, or stickers attached directly to shelf edges, rack frames, or shelf faces (not product display trays). Usually a long horizontal strip at shelf level with brand colors and logo.

# ### OFFER_RIBBON_STICKER (Class ID: 1027)
# A small sticker or tag placed on or near a product/bottle, typically showing a promotional offer, combo deal, or price. Often bow/ribbon-shaped, circular, or rectangular. Attached to individual product units or shelf labels.

# ### MT_RACK_DISPLAY (Class ID: 1087)
# A branded standalone rack or display structure specifically designed to hold and display Coca-Cola products. Often features brand graphics on side panels and a header board. Freestanding unit placed on the shop floor or shelf.

# ### SHOPPER_GATE (Class ID: 1092)
# A large entrance arch, gate, or doorway structure with Coca-Cola branding — used at store entrances, mall activations, or events. Walk-through arch format. Branded on the uprights and/or crossbar.

# ### LED_ELEMENT (Class ID: 1091)
# An illuminated or backlit/digital display panel with Coca-Cola branding. Glowing, lit, or screen-based. May be wall-mounted or shelf-mounted. Distinctly brighter than surrounding elements.

# ### SHELF_DISPLAY_MT (Class ID: 1064)
# A small, compact branded display unit placed ON a shelf (not freestanding on floor). Often a tray, mini-rack, or riser holding product units. Smaller footprint than MT_RACK_DISPLAY. Shelf-level placement.

# ### NECK_RINGER (Class ID: 1088)
# A small tag or card that hangs around the neck of a bottle, attached via a loop or ring. Typically circular or rectangular, shows brand messaging or offers. Fits around bottle neck opening area.

# ### BOX_DISPLAY (Class ID: 1089)
# A branded cardboard or rigid box structure used as a product display unit. May hold multiple product units together. Branded exterior panels. Can be placed on counters or shelves.

# ### COOLER_STRIPS (Class ID: 1090)
# Thin horizontal branding strips — usually a sequence of alternating brand logos — attached to the door edges, frame rails, or front face of a refrigerator/cooler unit. Narrower than full cooler wraps.

# ### FOAM_BANNER (Class ID: 1080)
# A thick, rigid banner made of foam board or PVC foam material. Sturdier and more rigid than regular printed banners. Usually displayed leaning against a wall or attached to a surface. May show campaign visuals.

# ---

# ## Similarity Resolution Logic (CRITICAL)

# When two classes could apply to the same object, resolve using:

# | Factor | What to check |
# |--------|--------------|
# | **Mounting type** | Hanging from ceiling → DANGLER; attached to shelf edge → WOBBLER; fixed to wall → POSTER |
# | **Size** | Small tag → WOBBLER / NECK_RINGER / OFFER_RIBBON_STICKER; large surface → WALL_BRANDING / FOAM_BANNER |
# | **Placement** | Ceiling → DANGLER / STREAMER; window glass → FRONT_WINDOW_BRANDING; shelf ON TOP → SHELF_DISPLAY_MT; shelf FACE → SHELF_RACK_BRANDING |
# | **Structure** | Freestanding rack → MT_RACK_DISPLAY; arch/gate → SHOPPER_GATE; illuminated → LED_ELEMENT |
# | **Rigidity** | Rigid foam/PVC → FOAM_BANNER; flexible thin strip → STREAMER; flexible shelf tag → WOBBLER |

# **Rule**: If still ambiguous after applying the above → pick the more conservative/specific class. If genuinely unresolvable → mark as "N".

# ---

# ## Confidence Scoring Guide (applies to "Y" detections only)

# | Score | Meaning |
# |-------|---------|
# | 85–100 | Element clearly visible, branding unmistakable, class is unambiguous |
# | 65–84 | Element partially visible or slightly ambiguous class, but best match is clear |
# | 40–64 | Element edge/corner visible or branding partially legible, but classification is reasonable |
# | Below 40 | Do NOT mark as "Y" — set result to "N" instead |

# ---

# ## Output Format (STRICT JSON ONLY)

# You MUST return all 19 classes in the detections array — one entry per class, regardless of whether it is present or not.

# {
#   "detections": [
#     {
#       "class_name": "<CLASS_NAME>",
#       "class_id": <CLASS_ID>,
#       "result": "Y",
#       "confidence": <integer 40-100>
#     },
#     {
#       "class_name": "<CLASS_NAME>",
#       "class_id": <CLASS_ID>,
#       "result": "N",
#       "confidence": 0
#     }
#   ]
# }

# **Rules:**
# - Output ONLY the JSON object. No explanation, no markdown, no preamble.
# - class_name must exactly match one of the 19 class names above (all caps with underscores).
# - class_id must be the corresponding integer ID from the class definitions above.
# - result must be exactly "Y" or "N" — no other values allowed.
# - confidence must be an integer 40–100 when result is "Y". Set confidence to 0 when result is "N".
# - ALL 19 classes must appear in the output — do not omit any class.
# - Sort detections by confidence descending (all "Y" entries first, then "N" entries).
# """

# # 📡 API CALL
# response = client.chat.completions.create(
#     model="gpt-4o",
#     temperature=0,
#     max_tokens=800,
#     response_format={"type": "json_object"},  # ✅ forces JSON
#     messages=[
#         {"role": "system", "content": PROMPT},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "Analyze this image carefully."},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}"
#                     }
#                 }
#             ],
#         },
#     ]
# )

# # 📊 PRINT RESULT
# print(response.choices[0].message.content)







"""
parse_trains.py
---------------
Extracts train_number, from_station, to_station from the
Indian Railways Train Number Index PDF and saves to a CSV.

The PDF has a two-column layout per page. Each row looks like:
    10001/10002  Balaghat  Jabalpur  Satpura  58

For each pair like 10001/10002:
  - Row 1: train=10001, from=Balaghat, to=Jabalpur   (as in PDF)
  - Row 2: train=10002, from=Jabalpur, to=Balaghat   (reversed)

Multi-line station names (e.g. "Sri Chhatrapati / Shahu Maharaj (T)")
are joined automatically using bounding-box grouping.

Requirements:
    pip install pdfplumber

Usage:
    python parse_trains.py                        # default paths
    python parse_trains.py input.pdf output.csv   # custom paths
"""

import sys
import re
import csv
import pdfplumber

# ---------------------------------------------------------------------------
# Column x-coordinate boundaries  (PDF points, from left edge of page)
# Calibrated so train-name and table-number columns are excluded.
# ---------------------------------------------------------------------------
L_TRAIN = (45,  90)     # left half  | train pair   e.g. "10001/10002"
L_FROM  = (90,  155)    # left half  | from station
L_TO    = (155, 217)    # left half  | to station   (train name starts ~217)

R_TRAIN = (290, 340)    # right half | train pair
R_FROM  = (335, 402)    # right half | from station
R_TO    = (402, 464)    # right half | to station   (train name starts ~464)

# Regex: valid train-number pair  e.g. "10001/10002"
PAIR_RE = re.compile(r'^\d{4,6}/\d{4,6}$')

# First-word tokens that identify header rows (skip as continuations)
SKIP_FIRST = {
    'Train', 'From', 'To', 'Name', 'No.', 'station',
    'Index', 'Number', 'TAG-13', '31-GAT',
}

# Month names — used to detect the calendar section at end of PDF
MONTH_NAMES = {
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def in_range(x, rng):
    return rng[0] <= x <= rng[1]


def group_by_row(words, tolerance=3):
    """Bucket words into rows by their vertical 'top' position."""
    rows = {}
    for w in words:
        key = round(w['top'] / tolerance) * tolerance
        rows.setdefault(key, []).append(w)
    return dict(sorted(rows.items()))


def collect(row_words, x_range):
    """Left-to-right text of words whose x0 falls within x_range."""
    ws = [w for w in row_words if in_range(w['x0'], x_range)]
    ws.sort(key=lambda w: w['x0'])
    return ' '.join(w['text'] for w in ws)


def is_header(text):
    return bool(text) and text.split()[0] in SKIP_FIRST


def is_calendar_junk(text):
    """
    True if text looks like calendar grid content:
    - contains a month name, OR
    - more than half the tokens are bare digits
    """
    if any(m in text for m in MONTH_NAMES):
        return True
    tokens = text.split()
    if len(tokens) < 3:
        return False
    digit_count = sum(1 for t in tokens if re.fullmatch(r'\d+', t))
    return digit_count / len(tokens) > 0.5


def clean_station(text):
    """
    Strip trailing calendar noise from a station name.
    Removes trailing pure-digit tokens and single capital-letter tokens
    (the day-of-week letters S M T W F that bleed from calendar grids).
    """
    tokens = text.split()
    while tokens:
        last = tokens[-1]
        if re.fullmatch(r'\d+', last):       # pure digit  e.g. "19"
            tokens.pop()
        elif re.fullmatch(r'[A-Z]', last):   # single cap  e.g. "S", "M", "T"
            tokens.pop()
        else:
            break
    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_half(rows_dict, train_range, from_range, to_range):
    """
    Parse one half (left or right) of a page.
    Returns list of (train_pair, from_station, to_station).
    Continuation rows (multi-line station names) are merged into the
    previous entry.
    """
    entries       = []
    pending_train = None
    pending_from  = ''
    pending_to    = ''

    for _top, wlist in rows_dict.items():
        t_text  = collect(wlist, train_range)
        f_text  = collect(wlist, from_range)
        to_text = collect(wlist, to_range)

        if PAIR_RE.match(t_text):
            # Flush previous entry
            if pending_train:
                entries.append((
                    pending_train,
                    clean_station(pending_from.strip()),
                    clean_station(pending_to.strip()),
                ))
            pending_train = t_text
            pending_from  = f_text
            pending_to    = to_text

        elif pending_train:
            # Possible continuation line (e.g. "Shahu Maharaj (T)")
            if (f_text
                    and not is_header(f_text)
                    and not is_calendar_junk(f_text)):
                pending_from = (pending_from + ' ' + f_text).strip()

            if (to_text
                    and not is_header(to_text)
                    and not is_calendar_junk(to_text)):
                pending_to = (pending_to + ' ' + to_text).strip()

    # Flush last entry
    if pending_train:
        entries.append((
            pending_train,
            clean_station(pending_from.strip()),
            clean_station(pending_to.strip()),
        ))

    return entries


# ---------------------------------------------------------------------------
# Entry expansion and CSV writing
# ---------------------------------------------------------------------------

def expand_entry(entry):
    """
    One PDF row → two CSV rows:
      forward : (first_no,  from, to)
      backward: (second_no, to,   from)
    """
    pair, from_st, to_st = entry
    no1, no2 = pair.split('/')
    return [
        (no1.strip(), from_st, to_st),
        (no2.strip(), to_st,   from_st),
    ]


def extract_trains(pdf_path):
    all_entries = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            rows  = group_by_row(words, tolerance=3)

            left_entries  = parse_half(rows, L_TRAIN, L_FROM, L_TO)
            right_entries = parse_half(rows, R_TRAIN, R_FROM, R_TO)

            all_entries.extend(left_entries)
            all_entries.extend(right_entries)

            print(f"  Page {page_num:2d}: "
                  f"{len(left_entries):3d} left + "
                  f"{len(right_entries):3d} right = "
                  f"{len(left_entries)+len(right_entries):3d} pairs")

    return all_entries


def write_csv(entries, out_path):
    seen = set()
    rows = []

    for entry in entries:
        for row in expand_entry(entry):
            if row[0] not in seen:
                seen.add(row[0])
                rows.append(row)

    rows.sort(key=lambda r: r[0])

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['train_number', 'from_station', 'to_station'])
        writer.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else 'Train_No-Index.pdf'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'trains.csv'

    print(f"Reading : {pdf_path}")
    entries = extract_trains(pdf_path)

    print(f"\nTotal pairs found : {len(entries)}")
    total = write_csv(entries, out_path)
    print(f"Total rows written: {total}  (each pair → 2 rows)")
    print(f"Output saved to   : {out_path}")