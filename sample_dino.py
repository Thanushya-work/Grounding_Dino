# # # # import os
# # # # import glob
# # # # import cv2
# # # # import torch
# # # # from ultralytics import YOLO

# # # # from groundingdino.util.inference import (
# # # #     load_model,
# # # #     predict,
# # # #     load_image,
# # # #     annotate
# # # # )

# # # # #########################################################
# # # # # CONFIG
# # # # #########################################################

# # # # INPUT_DIR = "shelf_images/shelf_images"          # folder containing images to test
# # # # OUTPUT_DIR = "output"         # folder where results will be saved
# # # # IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# # # # YOLO_MODEL = "data/availability_may_5th.pt"

# # # # GROUNDING_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# # # # GROUNDING_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

# # # # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # # # # GroundingDINO prompt — explicit categories of non-beverage items that
# # # # # make a cooler impure. Beverage bottles matched against YOLO's boxes
# # # # # are excluded; anything else detected here is treated as impure.
# # # # GROUNDING_PROMPT = (
# # # #     "cup . "
# # # #     "mug . "
# # # #     "glass . "
# # # #     "bowl . "
# # # #     "plate . "
# # # #     "spoon . "
# # # #     "fork . "
# # # #     "knife . "
# # # #     "plastic container . "
# # # #     "glass jar . "
# # # #     "steel container . "
# # # #     "steel utensil . "
# # # #     "plastic jar . "
# # # #     "lunch box . "
# # # #     "tiffin box . "
# # # #     "food packet . "
# # # #     "snack packet . "
# # # #     "chips packet . "
# # # #     "bread . "
# # # #     "cake . "
# # # #     "egg . "
# # # #     "fruit . "
# # # #     "vegetable . "
# # # #     "milk packet . "
# # # #     "curd cup . "
# # # #     "yogurt cup . "
# # # #     "paneer packet . "
# # # #     "ice cream . "
# # # #     "plastic bag . "
# # # #     "plastic cover . "
# # # #     "cloth . "
# # # #     "tissue . "
# # # #     "medicine . "
# # # #     "medicine bottle . "
# # # #     "cosmetic . "
# # # #     "mobile phone . "
# # # #     "wallet . "
# # # #     "keys . "
# # # #     "toy . "
# # # #     "box . "
# # # #     "container"
# # # # )

# # # # # Allowed: beverage bottles / soft drink bottles (glass, plastic, etc. —
# # # # # whatever the YOLO beverage model detects). Everything else is impure.
# # # # BEVERAGE_LABEL = "Beverage Bottle"
# # # # IMPURE_LABEL = "Impure Object"

# # # # #########################################################
# # # # # DRAWING HELPERS
# # # # #########################################################

# # # # def draw_label_box(img, box, text, color, font_scale=0.9, thickness=2):
# # # #     """Draw a bounding box with a filled label background and readable text."""
# # # #     x1, y1, x2, y2 = box

# # # #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

# # # #     (tw, th), baseline = cv2.getTextSize(
# # # #         text,
# # # #         cv2.FONT_HERSHEY_SIMPLEX,
# # # #         font_scale,
# # # #         thickness
# # # #     )

# # # #     text_y = max(th + 8, y1)

# # # #     cv2.rectangle(
# # # #         img,
# # # #         (x1, text_y - th - baseline - 4),
# # # #         (x1 + tw + 8, text_y + baseline),
# # # #         color,
# # # #         -1
# # # #     )

# # # #     cv2.putText(
# # # #         img,
# # # #         text,
# # # #         (x1 + 4, text_y),
# # # #         cv2.FONT_HERSHEY_SIMPLEX,
# # # #         font_scale,
# # # #         (255, 255, 255),
# # # #         thickness,
# # # #         cv2.LINE_AA
# # # #     )


# # # # def draw_banner(img, impure):
# # # #     """Draw the PURE / IMPURE result banner at the top-left of the image."""
# # # #     if impure:
# # # #         banner = "RESULT : IMPURE"
# # # #         banner_color = (0, 0, 255)
# # # #     else:
# # # #         banner = "RESULT : PURE"
# # # #         banner_color = (0, 180, 0)

# # # #     cv2.rectangle(img, (0, 0), (520, 55), banner_color, -1)

# # # #     cv2.putText(
# # # #         img,
# # # #         banner,
# # # #         (15, 38),
# # # #         cv2.FONT_HERSHEY_SIMPLEX,
# # # #         1.2,
# # # #         (255, 255, 255),
# # # #         3
# # # #     )


# # # # #########################################################
# # # # # IOU / CONTAINMENT
# # # # #########################################################

# # # # def iou(boxA, boxB):

# # # #     xA = max(boxA[0], boxB[0])
# # # #     yA = max(boxA[1], boxB[1])

# # # #     xB = min(boxA[2], boxB[2])
# # # #     yB = min(boxA[3], boxB[3])

# # # #     inter = max(0, xB - xA) * max(0, yB - yA)

# # # #     if inter == 0:
# # # #         return 0

# # # #     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
# # # #     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

# # # #     return inter / (areaA + areaB - inter)


# # # # def containment_ratio(boxA, boxB):
# # # #     """
# # # #     Fraction of the SMALLER box that lies inside the other box.
# # # #     Unlike IoU, this isn't penalized by a big size mismatch — so a
# # # #     tiny GroundingDINO box drawn on a bottle cap/label/reflection
# # # #     that sits inside a much larger YOLO beverage box still scores
# # # #     close to 1.0 here, even though its IoU would be near 0.
# # # #     """

# # # #     xA = max(boxA[0], boxB[0])
# # # #     yA = max(boxA[1], boxB[1])

# # # #     xB = min(boxA[2], boxB[2])
# # # #     yB = min(boxA[3], boxB[3])

# # # #     inter = max(0, xB - xA) * max(0, yB - yA)

# # # #     if inter == 0:
# # # #         return 0

# # # #     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
# # # #     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

# # # #     smaller_area = min(areaA, areaB)

# # # #     if smaller_area <= 0:
# # # #         return 0

# # # #     return inter / smaller_area


# # # # def overlaps_beverage(candidate_box, beverage_boxes, iou_thresh=0.5, containment_thresh=0.6):
# # # #     """
# # # #     True if candidate_box should be treated as "part of a beverage"
# # # #     (and therefore excluded from the impure list) — either because it
# # # #     substantially overlaps a beverage box (IoU), or because it is
# # # #     mostly contained inside one (e.g. a small GroundingDINO detection
# # # #     on a bottle cap/label sitting inside a full-bottle YOLO box).
# # # #     """

# # # #     for bbox in beverage_boxes:
# # # #         if iou(candidate_box, bbox) > iou_thresh:
# # # #             return True
# # # #         if containment_ratio(candidate_box, bbox) > containment_thresh:
# # # #             return True

# # # #     return False


# # # # #########################################################
# # # # # LOAD MODELS
# # # # #########################################################

# # # # print("Loading Beverage YOLO...")
# # # # beverage_model = YOLO(YOLO_MODEL)

# # # # print("Loading GroundingDINO...")
# # # # grounding_model = load_model(
# # # #     GROUNDING_CONFIG,
# # # #     GROUNDING_WEIGHTS
# # # # )

# # # # #########################################################
# # # # # PER-IMAGE PIPELINE
# # # # #########################################################

# # # # def process_image(image_path, out_dir):
# # # #     """
# # # #     Runs the full YOLO + GroundingDINO + SigLIP purity pipeline on a
# # # #     single image and saves the annotated visualization + raw
# # # #     GroundingDINO debug annotation into out_dir.

# # # #     Returns True if the image is classified IMPURE, False if PURE.
# # # #     """

# # # #     stem = os.path.splitext(os.path.basename(image_path))[0]

# # # #     image = cv2.imread(image_path)

# # # #     if image is None:
# # # #         print(f"  [SKIP] Could not read image: {image_path}")
# # # #         return None

# # # #     gd_image_source, gd_image_tensor = load_image(image_path)

# # # #     H, W = image.shape[:2]

# # # #     # This will accumulate all drawing (YOLO boxes, SigLIP boxes, banner)
# # # #     vis = image.copy()

# # # #     #####################################################
# # # #     # STEP 1 — Beverage Detection (YOLO)
# # # #     #####################################################

# # # #     yolo_result = beverage_model(image, verbose=False)[0]

# # # #     beverage_boxes = []

# # # #     for box in yolo_result.boxes:
# # # #         x1, y1, x2, y2 = map(int, box.xyxy[0])
# # # #         beverage_boxes.append([x1, y1, x2, y2])

# # # #     print(f"  Detected beverages: {len(beverage_boxes)}")

# # # #     for box in beverage_boxes:
# # # #         x1, y1, x2, y2 = box

# # # #         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

# # # #         cv2.putText(
# # # #             vis,
# # # #             BEVERAGE_LABEL,
# # # #             (x1, max(25, y1 - 5)),
# # # #             cv2.FONT_HERSHEY_SIMPLEX,
# # # #             0.7,
# # # #             (0, 255, 0),
# # # #             2
# # # #         )

# # # #     #####################################################
# # # #     # STEP 2 — GroundingDINO (targeted prompt)
# # # #     #####################################################

# # # #     boxes, logits, phrases = predict(
# # # #         model=grounding_model,
# # # #         image=gd_image_tensor,
# # # #         caption=GROUNDING_PROMPT,
# # # #         box_threshold=0.30,
# # # #         text_threshold=0.25,
# # # #         device=DEVICE
# # # #     )

# # # #     print(f"  Total GroundingDINO objects: {len(boxes)}")

# # # #     # Raw GroundingDINO debug annotation (optional)
# # # #     annotated_frame = annotate(
# # # #         image_source=gd_image_source,
# # # #         boxes=boxes,
# # # #         logits=logits,
# # # #         phrases=phrases
# # # #     )

# # # #     gd_debug_path = os.path.join(out_dir, f"{stem}_groundingdino.jpg")
# # # #     cv2.imwrite(gd_debug_path, annotated_frame)

# # # #     #####################################################
# # # #     # STEP 3 — Convert Relative Boxes -> Absolute
# # # #     #####################################################

# # # #     gd_boxes = []

# # # #     for b, phrase in zip(boxes, phrases):
# # # #         cx, cy, w, h = b.cpu().numpy()

# # # #         x1 = int((cx - w / 2) * W)
# # # #         y1 = int((cy - h / 2) * H)
# # # #         x2 = int((cx + w / 2) * W)
# # # #         y2 = int((cy + h / 2) * H)

# # # #         gd_boxes.append(([x1, y1, x2, y2], phrase))

# # # #     #####################################################
# # # #     # STEP 4 — Unknown objects (no overlap with YOLO beverages)
# # # #     #####################################################

# # # #     unknown = []

# # # #     for gbox, phrase in gd_boxes:

# # # #         if not overlaps_beverage(gbox, beverage_boxes):
# # # #             unknown.append((gbox, phrase))

# # # #     print(f"  Unknown objects: {len(unknown)}")

# # # #     #####################################################
# # # #     # STEP 5 — Any detected impure item (cup, jar, utensil, packet, etc.)
# # # #     #####################################################

# # # #     impure = False

# # # #     for i, (box, phrase) in enumerate(unknown):

# # # #         x1, y1, x2, y2 = box

# # # #         x1 = max(0, x1)
# # # #         y1 = max(0, y1)
# # # #         x2 = min(W, x2)
# # # #         y2 = min(H, y2)

# # # #         if x2 <= x1 or y2 <= y1:
# # # #             continue

# # # #         impure = True

# # # #         item_name = phrase.strip() if phrase and phrase.strip() else IMPURE_LABEL

# # # #         print(f"    Unknown {i+1}: {item_name} -> IMPURE")

# # # #         draw_label_box(vis, (x1, y1, x2, y2), item_name, (0, 0, 255))

# # # #     #####################################################
# # # #     # STEP 6 — Final banner + save
# # # #     #####################################################

# # # #     draw_banner(vis, impure)

# # # #     out_path = os.path.join(out_dir, f"{stem}_purity.jpg")
# # # #     cv2.imwrite(out_path, vis)

# # # #     print(f"  RESULT: {'IMPURE' if impure else 'PURE'}  ->  {out_path}")

# # # #     return impure


# # # # #########################################################
# # # # # MAIN — RUN OVER A FOLDER OF IMAGES
# # # # #########################################################

# # # # if __name__ == "__main__":

# # # #     os.makedirs(OUTPUT_DIR, exist_ok=True)

# # # #     image_paths = sorted(
# # # #         p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
# # # #         if p.lower().endswith(IMAGE_EXTENSIONS)
# # # #     )

# # # #     if not image_paths:
# # # #         raise FileNotFoundError(
# # # #             f"No images found in '{INPUT_DIR}' "
# # # #             f"(looked for extensions: {IMAGE_EXTENSIONS})"
# # # #         )

# # # #     print(f"\nFound {len(image_paths)} image(s) in '{INPUT_DIR}'\n")

# # # #     results = {}

# # # #     for path in image_paths:
# # # #         print(f"Processing: {path}")
# # # #         impure = process_image(path, OUTPUT_DIR)

# # # #         if impure is not None:
# # # #             results[path] = "IMPURE" if impure else "PURE"

# # # #         print()

# # # #     #####################################################
# # # #     # SUMMARY
# # # #     #####################################################

# # # #     print("=" * 50)
# # # #     print("SUMMARY")
# # # #     print("=" * 50)

# # # #     pure_count = sum(1 for v in results.values() if v == "PURE")
# # # #     impure_count = sum(1 for v in results.values() if v == "IMPURE")

# # # #     for path, verdict in results.items():
# # # #         print(f"{os.path.basename(path):40s} {verdict}")

# # # #     print("-" * 50)
# # # #     print(f"Total: {len(results)}   PURE: {pure_count}   IMPURE: {impure_count}")
# # # #     print(f"\nAll annotated images saved in: {OUTPUT_DIR}/")



# # # import os
# # # import glob
# # # import cv2
# # # import torch
# # # from ultralytics import YOLO

# # # from groundingdino.util.inference import (
# # #     load_model,
# # #     predict,
# # #     load_image,
# # #     annotate
# # # )

# # # #########################################################
# # # # CONFIG
# # # #########################################################

# # # INPUT_DIR = "downloaded_images"          # folder containing images to test
# # # OUTPUT_DIR = "output"         # folder where results will be saved
# # # IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# # # YOLO_MODEL = "data/availability_may_5th.pt"

# # # # Bottle-cap model. Detects caps and classifies them by brand (coca cola,
# # # # sprite, fanta, kinley, etc.) plus a catch-all "other(s)" class for caps
# # # # that don't match a known beverage brand.
# # # CAP_MODEL = "data/capmodel_june_26th.pt"

# # # GROUNDING_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# # # GROUNDING_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

# # # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # # # GroundingDINO prompt #1 — explicit categories of non-beverage items that
# # # # make a cooler impure. Gives nice, specific labels (e.g. "cup", "food
# # # # packet") when the object matches one of these known categories.
# # # GROUNDING_PROMPT_SPECIFIC = (
# # #     "cup . "
# # #     "mug . "
# # #     "glass . "
# # #     "bowl . "
# # #     "plate . "
# # #     "spoon . "
# # #     "fork . "
# # #     "knife . "
# # #     "plastic container . "
# # #     "glass jar . "
# # #     "steel container . "
# # #     "steel utensil . "
# # #     "plastic jar . "
# # #     "lunch box . "
# # #     "tiffin box . "
# # #     "food packet . "
# # #     "snack packet . "
# # #     "chips packet . "
# # #     "bread . "
# # #     "cake . "
# # #     "egg . "
# # #     "fruit . "
# # #     "vegetable . "
# # #     "milk packet . "
# # #     "curd cup . "
# # #     "yogurt cup . "
# # #     "paneer packet . "
# # #     "ice cream . "
# # #     "plastic bag . "
# # #     "plastic cover . "
# # #     "cloth . "
# # #     "tissue . "
# # #     "medicine . "
# # #     "medicine bottle . "
# # #     "cosmetic . "
# # #     "mobile phone . "
# # #     "wallet . "
# # #     "keys . "
# # #     "toy . "
# # #     "box . "
# # #     "container"
# # # )

# # # # GroundingDINO prompt #2 — generic, open-ended catch-all. This is what
# # # # gives us "anything other than a beverage bottle" coverage instead of
# # # # being limited to the fixed list above. Anything picked up here that
# # # # doesn't overlap a YOLO beverage box, and isn't already covered by the
# # # # specific prompt, gets flagged as a generic "item" impurity.
# # # GROUNDING_PROMPT_GENERIC = "object . item . thing . stuff"

# # # # Confidence thresholds per prompt. The generic prompt tends to be
# # # # noisier (it will happily fire on shelf edges, price tags, shadows,
# # # # etc.) so we keep its threshold a bit higher than the specific one.
# # # SPECIFIC_BOX_THRESHOLD = 0.30
# # # SPECIFIC_TEXT_THRESHOLD = 0.25

# # # GENERIC_BOX_THRESHOLD = 0.40
# # # GENERIC_TEXT_THRESHOLD = 0.30

# # # # Allowed: beverage bottles / soft drink bottles (glass, plastic, etc. —
# # # # whatever the YOLO beverage model detects). Everything else is impure.
# # # BEVERAGE_LABEL = "Beverage Bottle"
# # # IMPURE_LABEL = "Impure Object"
# # # GENERIC_IMPURE_LABEL = "Unidentified Item"

# # # # If a generic-prompt box overlaps a specific-prompt box by more than
# # # # this, we treat them as the same physical object and drop the generic
# # # # (less informative) one, keeping the specific label instead.
# # # DEDUPE_IOU_THRESH = 0.5
# # # DEDUPE_CONTAINMENT_THRESH = 0.6

# # # # A GroundingDINO detection is dropped (treated as "actually just a
# # # # beverage cap, not impure") if a KNOWN-BRAND cap (coca cola, sprite,
# # # # fanta, kinley, etc.) is found mostly inside it. Caps whose class name
# # # # starts with "other" (unrecognized brand) do NOT trigger this — those
# # # # stay flagged as impure, since we can't confirm they belong to an
# # # # allowed beverage.
# # # CAP_CONTAINMENT_THRESH = 0.5
# # # OTHER_CAP_PREFIX = "other"  # class names starting with this are NOT trusted brand caps
# # # CAP_LABEL_KNOWN = "Brand Cap"
# # # CAP_LABEL_OTHER = "Other Cap"

# # # #########################################################
# # # # DRAWING HELPERS
# # # #########################################################

# # # def draw_label_box(img, box, text, color, font_scale=0.9, thickness=2):
# # #     """Draw a bounding box with a filled label background and readable text."""
# # #     x1, y1, x2, y2 = box

# # #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

# # #     (tw, th), baseline = cv2.getTextSize(
# # #         text,
# # #         cv2.FONT_HERSHEY_SIMPLEX,
# # #         font_scale,
# # #         thickness
# # #     )

# # #     text_y = max(th + 8, y1)

# # #     cv2.rectangle(
# # #         img,
# # #         (x1, text_y - th - baseline - 4),
# # #         (x1 + tw + 8, text_y + baseline),
# # #         color,
# # #         -1
# # #     )

# # #     cv2.putText(
# # #         img,
# # #         text,
# # #         (x1 + 4, text_y),
# # #         cv2.FONT_HERSHEY_SIMPLEX,
# # #         font_scale,
# # #         (255, 255, 255),
# # #         thickness,
# # #         cv2.LINE_AA
# # #     )


# # # def draw_banner(img, impure):
# # #     """Draw the PURE / IMPURE result banner at the top-left of the image."""
# # #     if impure:
# # #         banner = "RESULT : IMPURE"
# # #         banner_color = (0, 0, 255)
# # #     else:
# # #         banner = "RESULT : PURE"
# # #         banner_color = (0, 180, 0)

# # #     cv2.rectangle(img, (0, 0), (520, 55), banner_color, -1)

# # #     cv2.putText(
# # #         img,
# # #         banner,
# # #         (15, 38),
# # #         cv2.FONT_HERSHEY_SIMPLEX,
# # #         1.2,
# # #         (255, 255, 255),
# # #         3
# # #     )


# # # #########################################################
# # # # IOU / CONTAINMENT
# # # #########################################################

# # # def iou(boxA, boxB):

# # #     xA = max(boxA[0], boxB[0])
# # #     yA = max(boxA[1], boxB[1])

# # #     xB = min(boxA[2], boxB[2])
# # #     yB = min(boxA[3], boxB[3])

# # #     inter = max(0, xB - xA) * max(0, yB - yA)

# # #     if inter == 0:
# # #         return 0

# # #     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
# # #     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

# # #     return inter / (areaA + areaB - inter)


# # # def containment_ratio(boxA, boxB):
# # #     """
# # #     Fraction of the SMALLER box that lies inside the other box.
# # #     Unlike IoU, this isn't penalized by a big size mismatch — so a
# # #     tiny GroundingDINO box drawn on a bottle cap/label/reflection
# # #     that sits inside a much larger YOLO beverage box still scores
# # #     close to 1.0 here, even though its IoU would be near 0.
# # #     """

# # #     xA = max(boxA[0], boxB[0])
# # #     yA = max(boxA[1], boxB[1])

# # #     xB = min(boxA[2], boxB[2])
# # #     yB = min(boxA[3], boxB[3])

# # #     inter = max(0, xB - xA) * max(0, yB - yA)

# # #     if inter == 0:
# # #         return 0

# # #     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
# # #     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

# # #     smaller_area = min(areaA, areaB)

# # #     if smaller_area <= 0:
# # #         return 0

# # #     return inter / smaller_area


# # # def overlaps_any(candidate_box, other_boxes, iou_thresh=0.5, containment_thresh=0.6):
# # #     """
# # #     True if candidate_box substantially overlaps ANY box in other_boxes,
# # #     either via IoU or containment (smaller box mostly inside larger box).
# # #     """

# # #     for bbox in other_boxes:
# # #         if iou(candidate_box, bbox) > iou_thresh:
# # #             return True
# # #         if containment_ratio(candidate_box, bbox) > containment_thresh:
# # #             return True

# # #     return False


# # # def contains_known_cap(candidate_box, known_cap_boxes, containment_thresh=0.5):
# # #     """
# # #     True if any KNOWN-BRAND cap box is mostly contained inside
# # #     candidate_box (i.e. the cap sits inside the GroundingDINO-detected
# # #     object). Caps are small, so we check containment (fraction of the
# # #     cap's own area that lies inside candidate_box) rather than IoU,
# # #     which would unfairly penalize the large size mismatch.
# # #     """

# # #     for cap_box in known_cap_boxes:
# # #         if containment_ratio(cap_box, candidate_box) > containment_thresh:
# # #             return True

# # #     return False


# # # #########################################################
# # # # LOAD MODELS
# # # #########################################################

# # # print("Loading Beverage YOLO...")
# # # beverage_model = YOLO(YOLO_MODEL)

# # # print("Loading Cap YOLO...")
# # # cap_model = YOLO(CAP_MODEL)

# # # print("Loading GroundingDINO...")
# # # grounding_model = load_model(
# # #     GROUNDING_CONFIG,
# # #     GROUNDING_WEIGHTS
# # # )

# # # #########################################################
# # # # HELPER — run GroundingDINO + convert to absolute boxes
# # # #########################################################

# # # def run_grounding(image_tensor, prompt, box_thresh, text_thresh, W, H):
# # #     """
# # #     Runs GroundingDINO with the given prompt/thresholds and returns a
# # #     list of (abs_box, phrase) tuples, where abs_box is [x1, y1, x2, y2]
# # #     in absolute pixel coordinates.
# # #     """

# # #     boxes, logits, phrases = predict(
# # #         model=grounding_model,
# # #         image=image_tensor,
# # #         caption=prompt,
# # #         box_threshold=box_thresh,
# # #         text_threshold=text_thresh,
# # #         device=DEVICE
# # #     )

# # #     abs_boxes = []

# # #     for b, phrase in zip(boxes, phrases):
# # #         cx, cy, w, h = b.cpu().numpy()

# # #         x1 = int((cx - w / 2) * W)
# # #         y1 = int((cy - h / 2) * H)
# # #         x2 = int((cx + w / 2) * W)
# # #         y2 = int((cy + h / 2) * H)

# # #         abs_boxes.append(([x1, y1, x2, y2], phrase))

# # #     return abs_boxes, boxes, logits, phrases


# # # #########################################################
# # # # PER-IMAGE PIPELINE
# # # #########################################################

# # # def process_image(image_path, out_dir):
# # #     """
# # #     Runs the full YOLO + GroundingDINO(specific+generic) + purity
# # #     pipeline on a single image and saves the annotated visualization +
# # #     raw GroundingDINO debug annotations into out_dir.

# # #     Returns True if the image is classified IMPURE, False if PURE.
# # #     """

# # #     stem = os.path.splitext(os.path.basename(image_path))[0]

# # #     image = cv2.imread(image_path)

# # #     if image is None:
# # #         print(f"  [SKIP] Could not read image: {image_path}")
# # #         return None

# # #     gd_image_source, gd_image_tensor = load_image(image_path)

# # #     H, W = image.shape[:2]

# # #     # This will accumulate all drawing (YOLO boxes, GD boxes, banner)
# # #     vis = image.copy()

# # #     #####################################################
# # #     # STEP 1 — Beverage Detection (YOLO)
# # #     #####################################################

# # #     yolo_result = beverage_model(image, verbose=False)[0]

# # #     beverage_boxes = []

# # #     for box in yolo_result.boxes:
# # #         x1, y1, x2, y2 = map(int, box.xyxy[0])
# # #         beverage_boxes.append([x1, y1, x2, y2])

# # #     print(f"  Detected beverages: {len(beverage_boxes)}")

# # #     for box in beverage_boxes:
# # #         x1, y1, x2, y2 = box

# # #         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

# # #         cv2.putText(
# # #             vis,
# # #             BEVERAGE_LABEL,
# # #             (x1, max(25, y1 - 5)),
# # #             cv2.FONT_HERSHEY_SIMPLEX,
# # #             0.7,
# # #             (0, 255, 0),
# # #             2
# # #         )

# # #     #####################################################
# # #     # STEP 1b — Bottle Cap Detection (YOLO)
# # #     #
# # #     # Known-brand caps (coca cola, sprite, fanta, kinley, ...) are used
# # #     # later to rescue GroundingDINO boxes that are actually just a
# # #     # branded bottle cap (false-positive impurity). Caps classified as
# # #     # "other(s)" (unrecognized brand) are tracked separately and are
# # #     # NOT allowed to rescue a box — an unrecognized cap keeps the
# # #     # object flagged as impure.
# # #     #####################################################

# # #     cap_result = cap_model(image, verbose=False)[0]

# # #     known_cap_boxes = []
# # #     other_cap_boxes = []

# # #     for box in cap_result.boxes:
# # #         x1, y1, x2, y2 = map(int, box.xyxy[0])
# # #         cls_id = int(box.cls[0])
# # #         cls_name = cap_model.names.get(cls_id, str(cls_id)) if isinstance(cap_model.names, dict) else cap_model.names[cls_id]

# # #         if cls_name.strip().lower().startswith(OTHER_CAP_PREFIX):
# # #             other_cap_boxes.append([x1, y1, x2, y2])
# # #             color = (0, 165, 255)   # orange — unrecognized cap
# # #             label = f"{CAP_LABEL_OTHER}: {cls_name}"
# # #         else:
# # #             known_cap_boxes.append([x1, y1, x2, y2])
# # #             color = (0, 255, 255)   # yellow — known brand cap
# # #             label = f"{CAP_LABEL_KNOWN}: {cls_name}"

# # #         cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
# # #         cv2.putText(
# # #             vis, label, (x1, max(15, y1 - 5)),
# # #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
# # #         )

# # #     print(f"  Detected caps: {len(known_cap_boxes)} known-brand, {len(other_cap_boxes)} other")

# # #     #####################################################
# # #     # STEP 2a — GroundingDINO (specific known-bad-category prompt)
# # #     #####################################################

# # #     specific_boxes, s_boxes_raw, s_logits, s_phrases = run_grounding(
# # #         gd_image_tensor,
# # #         GROUNDING_PROMPT_SPECIFIC,
# # #         SPECIFIC_BOX_THRESHOLD,
# # #         SPECIFIC_TEXT_THRESHOLD,
# # #         W, H
# # #     )

# # #     print(f"  Specific-prompt objects: {len(specific_boxes)}")

# # #     #####################################################
# # #     # STEP 2b — GroundingDINO (generic catch-all prompt)
# # #     #####################################################

# # #     generic_boxes, g_boxes_raw, g_logits, g_phrases = run_grounding(
# # #         gd_image_tensor,
# # #         GROUNDING_PROMPT_GENERIC,
# # #         GENERIC_BOX_THRESHOLD,
# # #         GENERIC_TEXT_THRESHOLD,
# # #         W, H
# # #     )

# # #     print(f"  Generic-prompt objects: {len(generic_boxes)}")

# # #     # Raw GroundingDINO debug annotations (optional, one per prompt)
# # #     specific_debug = annotate(
# # #         image_source=gd_image_source,
# # #         boxes=s_boxes_raw,
# # #         logits=s_logits,
# # #         phrases=s_phrases
# # #     )
# # #     cv2.imwrite(os.path.join(out_dir, f"{stem}_groundingdino_specific.jpg"), specific_debug)

# # #     generic_debug = annotate(
# # #         image_source=gd_image_source,
# # #         boxes=g_boxes_raw,
# # #         logits=g_logits,
# # #         phrases=g_phrases
# # #     )
# # #     cv2.imwrite(os.path.join(out_dir, f"{stem}_groundingdino_generic.jpg"), generic_debug)

# # #     #####################################################
# # #     # STEP 3 — Merge + de-duplicate the two GroundingDINO passes
# # #     #
# # #     # Keep every specific-prompt box as-is (it has a useful label).
# # #     # Add generic-prompt boxes ONLY if they don't already overlap a
# # #     # specific-prompt box (otherwise they're the same physical object,
# # #     # just re-detected under a vague label like "object").
# # #     #####################################################

# # #     specific_only_boxes = [b for b, _ in specific_boxes]

# # #     merged_boxes = list(specific_boxes)  # (box, phrase)

# # #     for gbox, gphrase in generic_boxes:
# # #         if not overlaps_any(
# # #             gbox,
# # #             specific_only_boxes,
# # #             DEDUPE_IOU_THRESH,
# # #             DEDUPE_CONTAINMENT_THRESH
# # #         ):
# # #             merged_boxes.append((gbox, gphrase if gphrase else GENERIC_IMPURE_LABEL))

# # #     print(f"  Merged (deduped) objects: {len(merged_boxes)}")

# # #     #####################################################
# # #     # STEP 4 — Unknown objects (no overlap with YOLO beverages)
# # #     #####################################################

# # #     unknown = []

# # #     for gbox, phrase in merged_boxes:

# # #         if not overlaps_any(gbox, beverage_boxes):
# # #             unknown.append((gbox, phrase))

# # #     print(f"  Unknown (non-beverage) objects: {len(unknown)}")

# # #     #####################################################
# # #     # STEP 5 — Any detected non-beverage item -> impure
# # #     #####################################################

# # #     impure = False

# # #     for i, (box, phrase) in enumerate(unknown):

# # #         x1, y1, x2, y2 = box

# # #         x1 = max(0, x1)
# # #         y1 = max(0, y1)
# # #         x2 = min(W, x2)
# # #         y2 = min(H, y2)

# # #         if x2 <= x1 or y2 <= y1:
# # #             continue

# # #         clipped_box = (x1, y1, x2, y2)

# # #         item_name = phrase.strip() if phrase and phrase.strip() else IMPURE_LABEL

# # #         # If a KNOWN-BRAND cap (coca cola, sprite, fanta, kinley, etc.)
# # #         # sits inside this box, treat it as a beverage false-positive
# # #         # and skip it — regardless of whether an "other" cap is also
# # #         # present. Only known-brand caps can rescue a box.
# # #         if contains_known_cap(clipped_box, known_cap_boxes, CAP_CONTAINMENT_THRESH):
# # #             print(f"    Unknown {i+1}: {item_name} -> RESCUED (known-brand cap found inside) -> ignored")
# # #             continue

# # #         impure = True

# # #         print(f"    Unknown {i+1}: {item_name} -> IMPURE")

# # #         draw_label_box(vis, (x1, y1, x2, y2), item_name, (0, 0, 255))

# # #     #####################################################
# # #     # STEP 6 — Final banner + save
# # #     #####################################################

# # #     draw_banner(vis, impure)

# # #     out_path = os.path.join(out_dir, f"{stem}_purity.jpg")
# # #     cv2.imwrite(out_path, vis)

# # #     print(f"  RESULT: {'IMPURE' if impure else 'PURE'}  ->  {out_path}")

# # #     return impure


# # # #########################################################
# # # # MAIN — RUN OVER A FOLDER OF IMAGES
# # # #########################################################

# # # if __name__ == "__main__":

# # #     os.makedirs(OUTPUT_DIR, exist_ok=True)

# # #     image_paths = sorted(
# # #         p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
# # #         if p.lower().endswith(IMAGE_EXTENSIONS)
# # #     )

# # #     if not image_paths:
# # #         raise FileNotFoundError(
# # #             f"No images found in '{INPUT_DIR}' "
# # #             f"(looked for extensions: {IMAGE_EXTENSIONS})"
# # #         )

# # #     print(f"\nFound {len(image_paths)} image(s) in '{INPUT_DIR}'\n")

# # #     results = {}

# # #     for path in image_paths:
# # #         print(f"Processing: {path}")
# # #         impure = process_image(path, OUTPUT_DIR)

# # #         if impure is not None:
# # #             results[path] = "IMPURE" if impure else "PURE"

# # #         print()

# # #     #####################################################
# # #     # SUMMARY
# # #     #####################################################

# # #     print("=" * 50)
# # #     print("SUMMARY")
# # #     print("=" * 50)

# # #     pure_count = sum(1 for v in results.values() if v == "PURE")
# # #     impure_count = sum(1 for v in results.values() if v == "IMPURE")

# # #     for path, verdict in results.items():
# # #         print(f"{os.path.basename(path):40s} {verdict}")

# # #     print("-" * 50)
# # #     print(f"Total: {len(results)}   PURE: {pure_count}   IMPURE: {impure_count}")
# # #     print(f"\nAll annotated images saved in: {OUTPUT_DIR}/")

# # import os
# # import glob
# # import cv2
# # import numpy as np
# # import torch
# # from ultralytics import YOLO

# # from groundingdino.util.inference import (
# #     load_model,
# #     predict,
# #     load_image,
# #     annotate
# # )

# # #########################################################
# # # CONFIG
# # #########################################################

# # INPUT_DIR = "downloaded_images"          # folder containing images to test
# # OUTPUT_DIR = "output"         # folder where results will be saved
# # IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# # YOLO_MODEL = "data/availability_may_5th.pt"

# # # Bottle-cap model. Detects caps and classifies them by brand (coca cola,
# # # sprite, fanta, kinley, etc.) plus a catch-all "other(s)" class for caps
# # # that don't match a known beverage brand.
# # CAP_MODEL = "data/capmodel_june_26th.pt"

# # GROUNDING_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# # GROUNDING_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # # GroundingDINO prompt #1 — explicit categories of non-beverage items that
# # # make a cooler impure. Gives nice, specific labels (e.g. "cup", "food
# # # packet") when the object matches one of these known categories.
# # GROUNDING_PROMPT_SPECIFIC = (
# #     "cup . "
# #     "mug . "
# #     "glass . "
# #     "bowl . "
# #     "plate . "
# #     "spoon . "
# #     "fork . "
# #     "knife . "
# #     "plastic container . "
# #     "glass jar . "
# #     "steel container . "
# #     "steel utensil . "
# #     "plastic jar . "
# #     "lunch box . "
# #     "tiffin box . "
# #     "food packet . "
# #     "snack packet . "
# #     "chips packet . "
# #     "bread . "
# #     "cake . "
# #     "egg . "
# #     "fruit . "
# #     "vegetable . "
# #     "milk packet . "
# #     "curd cup . "
# #     "yogurt cup . "
# #     "paneer packet . "
# #     "ice cream . "
# #     "plastic bag . "
# #     "plastic cover . "
# #     "cloth . "
# #     "tissue . "
# #     "medicine . "
# #     "medicine bottle . "
# #     "cosmetic . "
# #     "mobile phone . "
# #     "wallet . "
# #     "keys . "
# #     "toy . "
# #     "box . "
# #     "container"
# # )

# # # GroundingDINO prompt #2 — generic, open-ended catch-all. This is what
# # # gives us "anything other than a beverage bottle" coverage instead of
# # # being limited to the fixed list above. Anything picked up here that
# # # doesn't overlap a YOLO beverage box, and isn't already covered by the
# # # specific prompt, gets flagged as a generic "item" impurity.
# # GROUNDING_PROMPT_GENERIC = "object . item . thing . stuff"

# # # Confidence thresholds per prompt. The generic prompt tends to be
# # # noisier (it will happily fire on shelf edges, price tags, shadows,
# # # etc.) so we keep its threshold a bit higher than the specific one.
# # SPECIFIC_BOX_THRESHOLD = 0.33   # was 0.30 — small bump to cut weak noise
# # SPECIFIC_TEXT_THRESHOLD = 0.30  # was 0.25 — stricter text match = fewer
# #                                  # ambiguous labels like "glass jar" on bottles

# # GENERIC_BOX_THRESHOLD = 0.42
# # GENERIC_TEXT_THRESHOLD = 0.32

# # # Allowed: beverage bottles / soft drink bottles (glass, plastic, etc. —
# # # whatever the YOLO beverage model detects). Everything else is impure.
# # BEVERAGE_LABEL = "Beverage Bottle"
# # IMPURE_LABEL = "Impure Object"
# # GENERIC_IMPURE_LABEL = "Unidentified Item"

# # # If a generic-prompt box overlaps a specific-prompt box by more than
# # # this, we treat them as the same physical object and drop the generic
# # # (less informative) one, keeping the specific label instead.
# # DEDUPE_IOU_THRESH = 0.5
# # DEDUPE_CONTAINMENT_THRESH = 0.6

# # # A GroundingDINO detection is dropped (treated as "actually just a
# # # beverage cap, not impure") if a KNOWN-BRAND cap (coca cola, sprite,
# # # fanta, kinley, etc.) is found mostly inside it. Caps whose class name
# # # starts with "other" (unrecognized brand) do NOT trigger this — those
# # # stay flagged as impure, since we can't confirm they belong to an
# # # allowed beverage.
# # #
# # # FIX: lowered from 0.5 -> 0.35. GroundingDINO regularly draws its
# # # "glass jar" / "medicine bottle" boxes much taller than the actual
# # # bottle (extending above the cap or below the shelf), so a tight cap
# # # box sitting only in the top slice of a tall GD box was never clearing
# # # 0.5 containment even though it's obviously the same physical bottle.
# # CAP_CONTAINMENT_THRESH = 0.35
# # OTHER_CAP_PREFIX = "other"  # class names starting with this are NOT trusted brand caps
# # CAP_LABEL_KNOWN = "Brand Cap"
# # CAP_LABEL_OTHER = "Other Cap"

# # # FIX: also lower the beverage-box exclusion containment threshold for
# # # the same reason — a bottle's real YOLO box is tight, but GD's "glass
# # # jar"/"plastic jar" box on the same bottle is often loose/oversized.
# # BEVERAGE_CONTAINMENT_THRESH = 0.4  # was implicitly 0.6 via overlaps_any default
# # BEVERAGE_IOU_THRESH = 0.5

# # #########################################################
# # # FIX — Signage / branding-strip filter
# # #
# # # The printed Coca-Cola / Sprite / Fanta branding strip at the top or
# # # bottom of many shelves gets misread by GroundingDINO's specific
# # # prompt as a "lunch box" / "tiffin box" / "box" extremely
# # # consistently. It's flat printed signage, not a real object, and it's
# # # always a very wide, comparatively short horizontal band spanning
# # # most of the image width. We drop any detection matching that shape,
# # # regardless of label, before it ever reaches the impurity list.
# # #########################################################

# # SIGNAGE_MIN_WIDTH_FRAC = 0.55   # box spans at least 55% of image width
# # SIGNAGE_MAX_HEIGHT_FRAC = 0.18  # and is short vertically (a "strip")


# # def looks_like_signage_strip(box, W, H):
# #     x1, y1, x2, y2 = box
# #     box_w = x2 - x1
# #     box_h = y2 - y1

# #     if box_w <= 0 or box_h <= 0:
# #         return False

# #     wide_enough = box_w >= SIGNAGE_MIN_WIDTH_FRAC * W
# #     short_enough = box_h <= SIGNAGE_MAX_HEIGHT_FRAC * H

# #     # Signage strips sit right at the very top or very bottom of frame.
# #     near_top = y1 <= 0.12 * H
# #     near_bottom = y2 >= 0.88 * H

# #     return wide_enough and short_enough and (near_top or near_bottom)


# # #########################################################
# # # FIX — Low-contrast / low-variance region filter
# # #
# # # Vague terms like "steel container" / "container" occasionally fire
# # # on dark, near-uniform background — reflections in an empty shelf
# # # corner, compressor housing, shadow under a rack — where there's no
# # # real object with visible structure. Real objects (bottles, packets,
# # # bags) have noticeably more local pixel variance than a flat dark
# # # patch. We use this as a cheap sanity check restricted to the vaguest
# # # category labels, so it doesn't suppress genuinely low-contrast real
# # # items like a dark glass bottle (those are already excluded via the
# # # beverage-box check upstream).
# # #########################################################

# # VAGUE_CONTAINER_TERMS = ("container", "steel container", "box")
# # MIN_REGION_STD = 12.0  # pixel intensity std-dev threshold


# # def is_low_contrast_region(image, box):
# #     x1, y1, x2, y2 = box
# #     h, w = image.shape[:2]
# #     x1 = max(0, min(w - 1, x1))
# #     x2 = max(0, min(w, x2))
# #     y1 = max(0, min(h - 1, y1))
# #     y2 = max(0, min(h, y2))

# #     if x2 <= x1 or y2 <= y1:
# #         return True

# #     crop = image[y1:y2, x1:x2]
# #     if crop.size == 0:
# #         return True

# #     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# #     return float(gray.std()) < MIN_REGION_STD


# # #########################################################
# # # DRAWING HELPERS
# # #########################################################

# # def draw_label_box(img, box, text, color, font_scale=0.9, thickness=2):
# #     """Draw a bounding box with a filled label background and readable text."""
# #     x1, y1, x2, y2 = box

# #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

# #     (tw, th), baseline = cv2.getTextSize(
# #         text,
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         font_scale,
# #         thickness
# #     )

# #     text_y = max(th + 8, y1)

# #     cv2.rectangle(
# #         img,
# #         (x1, text_y - th - baseline - 4),
# #         (x1 + tw + 8, text_y + baseline),
# #         color,
# #         -1
# #     )

# #     cv2.putText(
# #         img,
# #         text,
# #         (x1 + 4, text_y),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         font_scale,
# #         (255, 255, 255),
# #         thickness,
# #         cv2.LINE_AA
# #     )


# # def draw_banner(img, impure):
# #     """Draw the PURE / IMPURE result banner at the top-left of the image."""
# #     if impure:
# #         banner = "RESULT : IMPURE"
# #         banner_color = (0, 0, 255)
# #     else:
# #         banner = "RESULT : PURE"
# #         banner_color = (0, 180, 0)

# #     cv2.rectangle(img, (0, 0), (520, 55), banner_color, -1)

# #     cv2.putText(
# #         img,
# #         banner,
# #         (15, 38),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         1.2,
# #         (255, 255, 255),
# #         3
# #     )


# # #########################################################
# # # IOU / CONTAINMENT
# # #########################################################

# # def iou(boxA, boxB):

# #     xA = max(boxA[0], boxB[0])
# #     yA = max(boxA[1], boxB[1])

# #     xB = min(boxA[2], boxB[2])
# #     yB = min(boxA[3], boxB[3])

# #     inter = max(0, xB - xA) * max(0, yB - yA)

# #     if inter == 0:
# #         return 0

# #     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
# #     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

# #     return inter / (areaA + areaB - inter)


# # def containment_ratio(boxA, boxB):
# #     """
# #     Fraction of the SMALLER box that lies inside the other box.
# #     Unlike IoU, this isn't penalized by a big size mismatch — so a
# #     tiny GroundingDINO box drawn on a bottle cap/label/reflection
# #     that sits inside a much larger YOLO beverage box still scores
# #     close to 1.0 here, even though its IoU would be near 0.
# #     """

# #     xA = max(boxA[0], boxB[0])
# #     yA = max(boxA[1], boxB[1])

# #     xB = min(boxA[2], boxB[2])
# #     yB = min(boxA[3], boxB[3])

# #     inter = max(0, xB - xA) * max(0, yB - yA)

# #     if inter == 0:
# #         return 0

# #     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
# #     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

# #     smaller_area = min(areaA, areaB)

# #     if smaller_area <= 0:
# #         return 0

# #     return inter / smaller_area


# # def overlaps_any(candidate_box, other_boxes, iou_thresh=0.5, containment_thresh=0.6):
# #     """
# #     True if candidate_box substantially overlaps ANY box in other_boxes,
# #     either via IoU or containment (smaller box mostly inside larger box).
# #     """

# #     for bbox in other_boxes:
# #         if iou(candidate_box, bbox) > iou_thresh:
# #             return True
# #         if containment_ratio(candidate_box, bbox) > containment_thresh:
# #             return True

# #     return False


# # def contains_known_cap(candidate_box, known_cap_boxes, containment_thresh=0.5):
# #     """
# #     True if any KNOWN-BRAND cap box is mostly contained inside
# #     candidate_box (i.e. the cap sits inside the GroundingDINO-detected
# #     object). Caps are small, so we check containment (fraction of the
# #     cap's own area that lies inside candidate_box) rather than IoU,
# #     which would unfairly penalize the large size mismatch.
# #     """

# #     for cap_box in known_cap_boxes:
# #         if containment_ratio(cap_box, candidate_box) > containment_thresh:
# #             return True

# #     return False


# # #########################################################
# # # LOAD MODELS
# # #########################################################

# # print("Loading Beverage YOLO...")
# # beverage_model = YOLO(YOLO_MODEL)

# # print("Loading Cap YOLO...")
# # cap_model = YOLO(CAP_MODEL)

# # print("Loading GroundingDINO...")
# # grounding_model = load_model(
# #     GROUNDING_CONFIG,
# #     GROUNDING_WEIGHTS
# # )

# # #########################################################
# # # HELPER — run GroundingDINO + convert to absolute boxes
# # #########################################################

# # def run_grounding(image_tensor, prompt, box_thresh, text_thresh, W, H):
# #     """
# #     Runs GroundingDINO with the given prompt/thresholds and returns a
# #     list of (abs_box, phrase) tuples, where abs_box is [x1, y1, x2, y2]
# #     in absolute pixel coordinates.
# #     """

# #     boxes, logits, phrases = predict(
# #         model=grounding_model,
# #         image=image_tensor,
# #         caption=prompt,
# #         box_threshold=box_thresh,
# #         text_threshold=text_thresh,
# #         device=DEVICE
# #     )

# #     abs_boxes = []

# #     for b, phrase in zip(boxes, phrases):
# #         cx, cy, w, h = b.cpu().numpy()

# #         x1 = int((cx - w / 2) * W)
# #         y1 = int((cy - h / 2) * H)
# #         x2 = int((cx + w / 2) * W)
# #         y2 = int((cy + h / 2) * H)

# #         abs_boxes.append(([x1, y1, x2, y2], phrase))

# #     return abs_boxes, boxes, logits, phrases


# # #########################################################
# # # PER-IMAGE PIPELINE
# # #########################################################

# # def process_image(image_path, out_dir):
# #     """
# #     Runs the full YOLO + GroundingDINO(specific+generic) + purity
# #     pipeline on a single image and saves the annotated visualization +
# #     raw GroundingDINO debug annotations into out_dir.

# #     Returns True if the image is classified IMPURE, False if PURE.
# #     """

# #     stem = os.path.splitext(os.path.basename(image_path))[0]

# #     image = cv2.imread(image_path)

# #     if image is None:
# #         print(f"  [SKIP] Could not read image: {image_path}")
# #         return None

# #     gd_image_source, gd_image_tensor = load_image(image_path)

# #     H, W = image.shape[:2]

# #     # This will accumulate all drawing (YOLO boxes, GD boxes, banner)
# #     vis = image.copy()

# #     #####################################################
# #     # STEP 1 — Beverage Detection (YOLO)
# #     #####################################################

# #     yolo_result = beverage_model(image, verbose=False)[0]

# #     beverage_boxes = []

# #     for box in yolo_result.boxes:
# #         x1, y1, x2, y2 = map(int, box.xyxy[0])
# #         beverage_boxes.append([x1, y1, x2, y2])

# #     print(f"  Detected beverages: {len(beverage_boxes)}")

# #     for box in beverage_boxes:
# #         x1, y1, x2, y2 = box

# #         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

# #         cv2.putText(
# #             vis,
# #             BEVERAGE_LABEL,
# #             (x1, max(25, y1 - 5)),
# #             cv2.FONT_HERSHEY_SIMPLEX,
# #             0.7,
# #             (0, 255, 0),
# #             2
# #         )

# #     #####################################################
# #     # STEP 1b — Bottle Cap Detection (YOLO)
# #     #
# #     # Known-brand caps (coca cola, sprite, fanta, kinley, ...) are used
# #     # later to rescue GroundingDINO boxes that are actually just a
# #     # branded bottle cap (false-positive impurity). Caps classified as
# #     # "other(s)" (unrecognized brand) are tracked separately and are
# #     # NOT allowed to rescue a box — an unrecognized cap keeps the
# #     # object flagged as impure.
# #     #####################################################

# #     cap_result = cap_model(image, verbose=False)[0]

# #     known_cap_boxes = []
# #     other_cap_boxes = []

# #     for box in cap_result.boxes:
# #         x1, y1, x2, y2 = map(int, box.xyxy[0])
# #         cls_id = int(box.cls[0])
# #         cls_name = cap_model.names.get(cls_id, str(cls_id)) if isinstance(cap_model.names, dict) else cap_model.names[cls_id]

# #         if cls_name.strip().lower().startswith(OTHER_CAP_PREFIX):
# #             other_cap_boxes.append([x1, y1, x2, y2])
# #             color = (0, 165, 255)   # orange — unrecognized cap
# #             label = f"{CAP_LABEL_OTHER}: {cls_name}"
# #         else:
# #             known_cap_boxes.append([x1, y1, x2, y2])
# #             color = (0, 255, 255)   # yellow — known brand cap
# #             label = f"{CAP_LABEL_KNOWN}: {cls_name}"

# #         cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
# #         cv2.putText(
# #             vis, label, (x1, max(15, y1 - 5)),
# #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
# #         )

# #     print(f"  Detected caps: {len(known_cap_boxes)} known-brand, {len(other_cap_boxes)} other")

# #     #####################################################
# #     # STEP 2a — GroundingDINO (specific known-bad-category prompt)
# #     #####################################################

# #     specific_boxes, s_boxes_raw, s_logits, s_phrases = run_grounding(
# #         gd_image_tensor,
# #         GROUNDING_PROMPT_SPECIFIC,
# #         SPECIFIC_BOX_THRESHOLD,
# #         SPECIFIC_TEXT_THRESHOLD,
# #         W, H
# #     )

# #     print(f"  Specific-prompt objects (raw): {len(specific_boxes)}")

# #     #####################################################
# #     # STEP 2b — GroundingDINO (generic catch-all prompt)
# #     #####################################################

# #     generic_boxes, g_boxes_raw, g_logits, g_phrases = run_grounding(
# #         gd_image_tensor,
# #         GROUNDING_PROMPT_GENERIC,
# #         GENERIC_BOX_THRESHOLD,
# #         GENERIC_TEXT_THRESHOLD,
# #         W, H
# #     )

# #     print(f"  Generic-prompt objects (raw): {len(generic_boxes)}")

# #     # Raw GroundingDINO debug annotations (optional, one per prompt)
# #     specific_debug = annotate(
# #         image_source=gd_image_source,
# #         boxes=s_boxes_raw,
# #         logits=s_logits,
# #         phrases=s_phrases
# #     )
# #     cv2.imwrite(os.path.join(out_dir, f"{stem}_groundingdino_specific.jpg"), specific_debug)

# #     generic_debug = annotate(
# #         image_source=gd_image_source,
# #         boxes=g_boxes_raw,
# #         logits=g_logits,
# #         phrases=g_phrases
# #     )
# #     cv2.imwrite(os.path.join(out_dir, f"{stem}_groundingdino_generic.jpg"), generic_debug)

# #     #####################################################
# #     # STEP 2c — FIX: drop signage-strip false positives
# #     #
# #     # Applied to both prompts before anything else touches them, so a
# #     # "lunch box" / "tiffin box" / "box" hit on the printed branding
# #     # strip never makes it into the merge/exclusion pipeline at all.
# #     #####################################################

# #     def strip_signage(boxes_with_phrase):
# #         kept = []
# #         dropped = 0
# #         for gbox, phrase in boxes_with_phrase:
# #             if looks_like_signage_strip(gbox, W, H):
# #                 dropped += 1
# #                 continue
# #             kept.append((gbox, phrase))
# #         if dropped:
# #             print(f"    Dropped {dropped} signage-strip false positive(s)")
# #         return kept

# #     specific_boxes = strip_signage(specific_boxes)
# #     generic_boxes = strip_signage(generic_boxes)

# #     #####################################################
# #     # STEP 3 — Merge + de-duplicate the two GroundingDINO passes
# #     #
# #     # Keep every specific-prompt box as-is (it has a useful label).
# #     # Add generic-prompt boxes ONLY if they don't already overlap a
# #     # specific-prompt box (otherwise they're the same physical object,
# #     # just re-detected under a vague label like "object").
# #     #####################################################

# #     specific_only_boxes = [b for b, _ in specific_boxes]

# #     merged_boxes = list(specific_boxes)  # (box, phrase)

# #     for gbox, gphrase in generic_boxes:
# #         if not overlaps_any(
# #             gbox,
# #             specific_only_boxes,
# #             DEDUPE_IOU_THRESH,
# #             DEDUPE_CONTAINMENT_THRESH
# #         ):
# #             merged_boxes.append((gbox, gphrase if gphrase else GENERIC_IMPURE_LABEL))

# #     print(f"  Merged (deduped) objects: {len(merged_boxes)}")

# #     #####################################################
# #     # STEP 4 — Unknown objects (no overlap with YOLO beverages)
# #     #
# #     # FIX: use the looser BEVERAGE_* thresholds here instead of the
# #     # overlaps_any() defaults, since GD's box on a bottle is often
# #     # drawn much larger/looser than YOLO's tight beverage box.
# #     #####################################################

# #     unknown = []

# #     for gbox, phrase in merged_boxes:

# #         if not overlaps_any(
# #             gbox,
# #             beverage_boxes,
# #             BEVERAGE_IOU_THRESH,
# #             BEVERAGE_CONTAINMENT_THRESH
# #         ):
# #             unknown.append((gbox, phrase))

# #     print(f"  Unknown (non-beverage) objects: {len(unknown)}")

# #     #####################################################
# #     # STEP 5 — Any detected non-beverage item -> impure
# #     #####################################################

# #     impure = False

# #     for i, (box, phrase) in enumerate(unknown):

# #         x1, y1, x2, y2 = box

# #         x1 = max(0, x1)
# #         y1 = max(0, y1)
# #         x2 = min(W, x2)
# #         y2 = min(H, y2)

# #         if x2 <= x1 or y2 <= y1:
# #             continue

# #         clipped_box = (x1, y1, x2, y2)

# #         item_name = phrase.strip() if phrase and phrase.strip() else IMPURE_LABEL

# #         # If a KNOWN-BRAND cap (coca cola, sprite, fanta, kinley, etc.)
# #         # sits inside this box, treat it as a beverage false-positive
# #         # and skip it — regardless of whether an "other" cap is also
# #         # present. Only known-brand caps can rescue a box.
# #         if contains_known_cap(clipped_box, known_cap_boxes, CAP_CONTAINMENT_THRESH):
# #             print(f"    Unknown {i+1}: {item_name} -> RESCUED (known-brand cap found inside) -> ignored")
# #             continue

# #         # FIX: vague "container"/"box" style labels on a near-uniform,
# #         # low-contrast patch (dark shelf corner, compressor housing,
# #         # reflection) are almost never a real object — drop them.
# #         lowered_name = item_name.lower()
# #         if any(term in lowered_name for term in VAGUE_CONTAINER_TERMS):
# #             if is_low_contrast_region(image, clipped_box):
# #                 print(f"    Unknown {i+1}: {item_name} -> DROPPED (low-contrast/no-object region) -> ignored")
# #                 continue

# #         impure = True

# #         print(f"    Unknown {i+1}: {item_name} -> IMPURE")

# #         draw_label_box(vis, (x1, y1, x2, y2), item_name, (0, 0, 255))

# #     #####################################################
# #     # STEP 6 — Final banner + save
# #     #####################################################

# #     draw_banner(vis, impure)

# #     out_path = os.path.join(out_dir, f"{stem}_purity.jpg")
# #     cv2.imwrite(out_path, vis)

# #     print(f"  RESULT: {'IMPURE' if impure else 'PURE'}  ->  {out_path}")

# #     return impure


# # #########################################################
# # # MAIN — RUN OVER A FOLDER OF IMAGES
# # #########################################################

# # if __name__ == "__main__":

# #     os.makedirs(OUTPUT_DIR, exist_ok=True)

# #     image_paths = sorted(
# #         p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
# #         if p.lower().endswith(IMAGE_EXTENSIONS)
# #     )

# #     if not image_paths:
# #         raise FileNotFoundError(
# #             f"No images found in '{INPUT_DIR}' "
# #             f"(looked for extensions: {IMAGE_EXTENSIONS})"
# #         )

# #     print(f"\nFound {len(image_paths)} image(s) in '{INPUT_DIR}'\n")

# #     results = {}

# #     for path in image_paths:
# #         print(f"Processing: {path}")
# #         impure = process_image(path, OUTPUT_DIR)

# #         if impure is not None:
# #             results[path] = "IMPURE" if impure else "PURE"

# #         print()

# #     #####################################################
# #     # SUMMARY
# #     #####################################################

# #     print("=" * 50)
# #     print("SUMMARY")
# #     print("=" * 50)

# #     pure_count = sum(1 for v in results.values() if v == "PURE")
# #     impure_count = sum(1 for v in results.values() if v == "IMPURE")

# #     for path, verdict in results.items():
# #         print(f"{os.path.basename(path):40s} {verdict}")

# #     print("-" * 50)
# #     print(f"Total: {len(results)}   PURE: {pure_count}   IMPURE: {impure_count}")
# #     print(f"\nAll annotated images saved in: {OUTPUT_DIR}/")


# import os
# import glob
# import cv2
# import numpy as np
# import torch
# from ultralytics import YOLO

# from groundingdino.util.inference import (
#     load_model,
#     predict,
#     load_image,
#     annotate
# )

# #########################################################
# # CONFIG
# #########################################################

# INPUT_DIR = "downloaded_images"          # folder containing images to test
# OUTPUT_DIR = "output"         # folder where results will be saved
# IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# YOLO_MODEL = "data/availability_may_5th.pt"

# # Bottle-cap model. Detects caps and classifies them by brand (coca cola,
# # sprite, fanta, kinley, etc.) plus a catch-all "other(s)" class for caps
# # that don't match a known beverage brand.
# CAP_MODEL = "data/capmodel_june_26th.pt"

# GROUNDING_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# GROUNDING_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # GroundingDINO prompt #1 — explicit categories of non-beverage items that
# # make a cooler impure. Gives nice, specific labels (e.g. "cup", "food
# # packet") when the object matches one of these known categories.
# GROUNDING_PROMPT_SPECIFIC = (
#     "cup . "
#     "mug . "
#     "glass . "
#     "bowl . "
#     "plate . "
#     "spoon . "
#     "fork . "
#     "knife . "
#     "plastic container . "
#     "glass jar . "
#     "steel container . "
#     "steel utensil . "
#     "plastic jar . "
#     "lunch box . "
#     "tiffin box . "
#     "food packet . "
#     "snack packet . "
#     "chips packet . "
#     "bread . "
#     "cake . "
#     "egg . "
#     "fruit . "
#     "vegetable . "
#     "milk packet . "
#     "curd cup . "
#     "yogurt cup . "
#     "paneer packet . "
#     "ice cream . "
#     "plastic bag . "
#     "plastic cover . "
#     "cloth . "
#     "tissue . "
#     "medicine . "
#     "medicine bottle . "
#     "cosmetic . "
#     "mobile phone . "
#     "wallet . "
#     "keys . "
#     "toy . "
#     "box . "
#     "container"
# )

# # GroundingDINO prompt #2 — generic, open-ended catch-all. This is what
# # gives us "anything other than a beverage bottle" coverage instead of
# # being limited to the fixed list above. Anything picked up here that
# # doesn't overlap a YOLO beverage box, and isn't already covered by the
# # specific prompt, gets flagged as a generic "item" impurity.
# GROUNDING_PROMPT_GENERIC = "object . item . thing . stuff"

# # Confidence thresholds per prompt. The generic prompt tends to be
# # noisier (it will happily fire on shelf edges, price tags, shadows,
# # etc.) so we keep its threshold a bit higher than the specific one.
# SPECIFIC_BOX_THRESHOLD = 0.75   # lowered from 0.33: the signage-strip,
#                                  # full-frame, and low-contrast filters
#                                  # below now catch the noise this used
#                                  # to suppress, so we can afford a lower
#                                  # bar and keep marginal true positives
#                                  # like "fruit 0.34" / "medicine bottle 0.34"
# SPECIFIC_TEXT_THRESHOLD = 0.25  # back to original — same reasoning

# GENERIC_BOX_THRESHOLD = 0.38    # still higher than specific since the
#                                  # generic prompt ("object . item . thing")
#                                  # is inherently vaguer and noisier
# GENERIC_TEXT_THRESHOLD = 0.28

# # Allowed: beverage bottles / soft drink bottles (glass, plastic, etc. —
# # whatever the YOLO beverage model detects). Everything else is impure.
# BEVERAGE_LABEL = "Beverage Bottle"
# IMPURE_LABEL = "Impure Object"
# GENERIC_IMPURE_LABEL = "Unidentified Item"

# # If a generic-prompt box overlaps a specific-prompt box by more than
# # this, we treat them as the same physical object and drop the generic
# # (less informative) one, keeping the specific label instead.
# DEDUPE_IOU_THRESH = 0.5
# DEDUPE_CONTAINMENT_THRESH = 0.6

# # A GroundingDINO detection is dropped (treated as "actually just a
# # beverage cap, not impure") if a KNOWN-BRAND cap (coca cola, sprite,
# # fanta, kinley, etc.) is found mostly inside it. Caps whose class name
# # starts with "other" (unrecognized brand) do NOT trigger this — those
# # stay flagged as impure, since we can't confirm they belong to an
# # allowed beverage.
# #
# # FIX: lowered from 0.5 -> 0.35. GroundingDINO regularly draws its
# # "glass jar" / "medicine bottle" boxes much taller than the actual
# # bottle (extending above the cap or below the shelf), so a tight cap
# # box sitting only in the top slice of a tall GD box was never clearing
# # 0.5 containment even though it's obviously the same physical bottle.
# CAP_CONTAINMENT_THRESH = 0.35
# OTHER_CAP_PREFIX = "other"  # class names starting with this are NOT trusted brand caps
# CAP_LABEL_KNOWN = "Brand Cap"
# CAP_LABEL_OTHER = "Other Cap"

# # FIX: also lower the beverage-box exclusion containment threshold for
# # the same reason — a bottle's real YOLO box is tight, but GD's "glass
# # jar"/"plastic jar" box on the same bottle is often loose/oversized.
# BEVERAGE_CONTAINMENT_THRESH = 0.4  # was implicitly 0.6 via overlaps_any default
# BEVERAGE_IOU_THRESH = 0.5

# #########################################################
# # FIX — Signage / branding-strip filter
# #
# # The printed Coca-Cola / Sprite / Fanta branding strip at the top or
# # bottom of many shelves gets misread by GroundingDINO's specific
# # prompt as a "lunch box" / "tiffin box" / "box" extremely
# # consistently. It's flat printed signage, not a real object, and it's
# # always a very wide, comparatively short horizontal band spanning
# # most of the image width. We drop any detection matching that shape,
# # regardless of label, before it ever reaches the impurity list.
# #########################################################

# SIGNAGE_MIN_WIDTH_FRAC = 0.55   # full-width case: box spans most of the image
# SIGNAGE_MAX_HEIGHT_FRAC = 0.18  # and is short vertically (a "strip")

# # FIX: bottles/other objects often occlude most of the branding strip,
# # so GD only detects the small visible sliver of it — which can be far
# # narrower than 55% of the frame. That sliver still has the telltale
# # shape of a signage strip: very wide relative to its OWN height, and
# # flush against the top/bottom edge. This catches it regardless of how
# # much of the strip is actually visible.
# SIGNAGE_MIN_ASPECT_RATIO = 4.0   # width / height
# SIGNAGE_MIN_ABS_WIDTH_FRAC = 0.12  # still require some minimum real width
#                                     # so we don't misfire on small rescued
#                                     # items like a single cap or label


# def looks_like_signage_strip(box, W, H):
#     x1, y1, x2, y2 = box
#     box_w = x2 - x1
#     box_h = y2 - y1

#     if box_w <= 0 or box_h <= 0:
#         return False

#     near_top = y1 <= 0.12 * H
#     near_bottom = y2 >= 0.88 * H
#     near_edge = near_top or near_bottom

#     # Case 1: full/near-full width strip (unoccluded).
#     wide_enough = box_w >= SIGNAGE_MIN_WIDTH_FRAC * W
#     short_enough = box_h <= SIGNAGE_MAX_HEIGHT_FRAC * H
#     if wide_enough and short_enough and near_edge:
#         return True

#     # Case 2: partially occluded strip — only a sliver is visible, but
#     # that sliver is still a flat, wide-relative-to-itself band right
#     # at the frame edge.
#     aspect_ratio = box_w / box_h
#     has_min_width = box_w >= SIGNAGE_MIN_ABS_WIDTH_FRAC * W
#     if aspect_ratio >= SIGNAGE_MIN_ASPECT_RATIO and has_min_width and near_edge:
#         return True

#     return False


# #########################################################
# # FIX — Degenerate full-frame box filter
# #
# # GroundingDINO occasionally returns a box that covers nearly the
# # entire image for a weakly-matched phrase (e.g. "lunch box" at ~0.48
# # confidence spanning almost the whole shelf, top-left to
# # bottom-right). This isn't a real localized object — it's a
# # degenerate detection — and it's a different shape than the signage
# # strip (tall, not short), so it needs its own check based purely on
# # how much of the total frame area the box covers.
# #########################################################

# MAX_BOX_AREA_FRAC = 0.65  # drop any detection covering more than 65% of the frame


# def is_degenerate_fullframe_box(box, W, H):
#     x1, y1, x2, y2 = box
#     box_w = max(0, x2 - x1)
#     box_h = max(0, y2 - y1)
#     box_area = box_w * box_h
#     frame_area = W * H

#     if frame_area <= 0:
#         return False

#     return (box_area / frame_area) > MAX_BOX_AREA_FRAC


# #########################################################
# # FIX — Low-contrast / low-variance region filter
# #
# # Vague terms like "steel container" / "container" occasionally fire
# # on dark, near-uniform background — reflections in an empty shelf
# # corner, compressor housing, shadow under a rack — where there's no
# # real object with visible structure. Real objects (bottles, packets,
# # bags) have noticeably more local pixel variance than a flat dark
# # patch. We use this as a cheap sanity check restricted to the vaguest
# # category labels, so it doesn't suppress genuinely low-contrast real
# # items like a dark glass bottle (those are already excluded via the
# # beverage-box check upstream).
# #########################################################

# VAGUE_CONTAINER_TERMS = ("container", "steel container", "box")
# MIN_REGION_STD = 12.0  # pixel intensity std-dev threshold


# def is_low_contrast_region(image, box):
#     x1, y1, x2, y2 = box
#     h, w = image.shape[:2]
#     x1 = max(0, min(w - 1, x1))
#     x2 = max(0, min(w, x2))
#     y1 = max(0, min(h - 1, y1))
#     y2 = max(0, min(h, y2))

#     if x2 <= x1 or y2 <= y1:
#         return True

#     crop = image[y1:y2, x1:x2]
#     if crop.size == 0:
#         return True

#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     return float(gray.std()) < MIN_REGION_STD


# #########################################################
# # FIX — Low-light preprocessing for the beverage detector
# #
# # Some real bottles are being MISSED entirely by the beverage YOLO
# # model — not mislabeled, not a loose GD box, just never detected —
# # because they sit in dark/underexposed regions (shelf edges, gaps
# # between bottle rows, poor lighting). When that happens there's no
# # beverage box for the exclusion logic to compare against, so
# # GroundingDINO's label on that same dark bottle ("steel container",
# # "medicine bottle") gets flagged as impure with nothing to rescue it.
# #
# # This is a detector-recall problem, not a box-geometry problem, so it
# # needs to be fixed before/around the beverage YOLO call itself. CLAHE
# # (contrast-limited adaptive histogram equalization) boosts local
# # contrast in dark regions without blowing out already-bright areas,
# # which tends to recover detail in underexposed bottles.
# #
# # NOTE: this is applied only to the copy of the image fed to the
# # beverage YOLO model — the original `image` used for GroundingDINO,
# # cap detection, and the final saved visualization is untouched.
# #########################################################

# def enhance_for_dark_regions(image):
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     l_enhanced = clahe.apply(l)

#     lab_enhanced = cv2.merge((l_enhanced, a, b))
#     return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

# def draw_label_box(img, box, text, color, font_scale=0.9, thickness=2):
#     """Draw a bounding box with a filled label background and readable text."""
#     x1, y1, x2, y2 = box

#     cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

#     (tw, th), baseline = cv2.getTextSize(
#         text,
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale,
#         thickness
#     )

#     text_y = max(th + 8, y1)

#     cv2.rectangle(
#         img,
#         (x1, text_y - th - baseline - 4),
#         (x1 + tw + 8, text_y + baseline),
#         color,
#         -1
#     )

#     cv2.putText(
#         img,
#         text,
#         (x1 + 4, text_y),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         font_scale,
#         (255, 255, 255),
#         thickness,
#         cv2.LINE_AA
#     )


# def draw_banner(img, impure):
#     """Draw the PURE / IMPURE result banner at the top-left of the image."""
#     if impure:
#         banner = "RESULT : IMPURE"
#         banner_color = (0, 0, 255)
#     else:
#         banner = "RESULT : PURE"
#         banner_color = (0, 180, 0)

#     cv2.rectangle(img, (0, 0), (520, 55), banner_color, -1)

#     cv2.putText(
#         img,
#         banner,
#         (15, 38),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.2,
#         (255, 255, 255),
#         3
#     )


# #########################################################
# # IOU / CONTAINMENT
# #########################################################

# def iou(boxA, boxB):

#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])

#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter = max(0, xB - xA) * max(0, yB - yA)

#     if inter == 0:
#         return 0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     return inter / (areaA + areaB - inter)


# def containment_ratio(boxA, boxB):
#     """
#     Fraction of the SMALLER box that lies inside the other box.
#     Unlike IoU, this isn't penalized by a big size mismatch — so a
#     tiny GroundingDINO box drawn on a bottle cap/label/reflection
#     that sits inside a much larger YOLO beverage box still scores
#     close to 1.0 here, even though its IoU would be near 0.
#     """

#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])

#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter = max(0, xB - xA) * max(0, yB - yA)

#     if inter == 0:
#         return 0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     smaller_area = min(areaA, areaB)

#     if smaller_area <= 0:
#         return 0

#     return inter / smaller_area


# def overlaps_any(candidate_box, other_boxes, iou_thresh=0.5, containment_thresh=0.6):
#     """
#     True if candidate_box substantially overlaps ANY box in other_boxes,
#     either via IoU or containment (smaller box mostly inside larger box).
#     """

#     for bbox in other_boxes:
#         if iou(candidate_box, bbox) > iou_thresh:
#             return True
#         if containment_ratio(candidate_box, bbox) > containment_thresh:
#             return True

#     return False


# def contains_known_cap(candidate_box, known_cap_boxes, containment_thresh=0.5):
#     """
#     True if any KNOWN-BRAND cap box is mostly contained inside
#     candidate_box (i.e. the cap sits inside the GroundingDINO-detected
#     object). Caps are small, so we check containment (fraction of the
#     cap's own area that lies inside candidate_box) rather than IoU,
#     which would unfairly penalize the large size mismatch.
#     """

#     for cap_box in known_cap_boxes:
#         if containment_ratio(cap_box, candidate_box) > containment_thresh:
#             return True

#     return False


# #########################################################
# # LOAD MODELS
# #########################################################

# print("Loading Beverage YOLO...")
# beverage_model = YOLO(YOLO_MODEL)

# print("Loading Cap YOLO...")
# cap_model = YOLO(CAP_MODEL)

# print("Loading GroundingDINO...")
# grounding_model = load_model(
#     GROUNDING_CONFIG,
#     GROUNDING_WEIGHTS
# )

# #########################################################
# # HELPER — run GroundingDINO + convert to absolute boxes
# #########################################################

# def run_grounding(image_tensor, prompt, box_thresh, text_thresh, W, H):
#     """
#     Runs GroundingDINO with the given prompt/thresholds and returns a
#     list of (abs_box, phrase) tuples, where abs_box is [x1, y1, x2, y2]
#     in absolute pixel coordinates.
#     """

#     boxes, logits, phrases = predict(
#         model=grounding_model,
#         image=image_tensor,
#         caption=prompt,
#         box_threshold=box_thresh,
#         text_threshold=text_thresh,
#         device=DEVICE
#     )

#     abs_boxes = []

#     for b, phrase in zip(boxes, phrases):
#         cx, cy, w, h = b.cpu().numpy()

#         x1 = int((cx - w / 2) * W)
#         y1 = int((cy - h / 2) * H)
#         x2 = int((cx + w / 2) * W)
#         y2 = int((cy + h / 2) * H)

#         abs_boxes.append(([x1, y1, x2, y2], phrase))

#     return abs_boxes, boxes, logits, phrases


# #########################################################
# # PER-IMAGE PIPELINE
# #########################################################

# def process_image(image_path, out_dir):
#     """
#     Runs the full YOLO + GroundingDINO(specific+generic) + purity
#     pipeline on a single image and saves the annotated visualization +
#     raw GroundingDINO debug annotations into out_dir.

#     Returns True if the image is classified IMPURE, False if PURE.
#     """

#     stem = os.path.splitext(os.path.basename(image_path))[0]

#     image = cv2.imread(image_path)

#     if image is None:
#         print(f"  [SKIP] Could not read image: {image_path}")
#         return None

#     gd_image_source, gd_image_tensor = load_image(image_path)

#     H, W = image.shape[:2]

#     # This will accumulate all drawing (YOLO boxes, GD boxes, banner)
#     vis = image.copy()

#     #####################################################
#     # STEP 1 — Beverage Detection (YOLO)
#     #
#     # FIX: run on a CLAHE-enhanced copy so dark/underexposed bottles
#     # (shelf edges, shadowed gaps) are more likely to be detected.
#     # `image` itself stays untouched for GD, cap detection, and the
#     # final saved visualization.
#     #####################################################

#     beverage_input = enhance_for_dark_regions(image)
#     yolo_result = beverage_model(beverage_input, verbose=False)[0]

#     beverage_boxes = []

#     for box in yolo_result.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         beverage_boxes.append([x1, y1, x2, y2])

#     print(f"  Detected beverages: {len(beverage_boxes)}")

#     for box in beverage_boxes:
#         x1, y1, x2, y2 = box

#         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         cv2.putText(
#             vis,
#             BEVERAGE_LABEL,
#             (x1, max(25, y1 - 5)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 0),
#             2
#         )

#     #####################################################
#     # STEP 1b — Bottle Cap Detection (YOLO)
#     #
#     # Known-brand caps (coca cola, sprite, fanta, kinley, ...) are used
#     # later to rescue GroundingDINO boxes that are actually just a
#     # branded bottle cap (false-positive impurity). Caps classified as
#     # "other(s)" (unrecognized brand) are tracked separately and are
#     # NOT allowed to rescue a box — an unrecognized cap keeps the
#     # object flagged as impure.
#     #####################################################

#     cap_result = cap_model(image, verbose=False)[0]

#     known_cap_boxes = []
#     other_cap_boxes = []

#     for box in cap_result.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls_id = int(box.cls[0])
#         cls_name = cap_model.names.get(cls_id, str(cls_id)) if isinstance(cap_model.names, dict) else cap_model.names[cls_id]

#         if cls_name.strip().lower().startswith(OTHER_CAP_PREFIX):
#             other_cap_boxes.append([x1, y1, x2, y2])
#             color = (0, 165, 255)   # orange — unrecognized cap
#             label = f"{CAP_LABEL_OTHER}: {cls_name}"
#         else:
#             known_cap_boxes.append([x1, y1, x2, y2])
#             color = (0, 255, 255)   # yellow — known brand cap
#             label = f"{CAP_LABEL_KNOWN}: {cls_name}"

#         cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(
#             vis, label, (x1, max(15, y1 - 5)),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
#         )

#     print(f"  Detected caps: {len(known_cap_boxes)} known-brand, {len(other_cap_boxes)} other")

#     #####################################################
#     # STEP 2a — GroundingDINO (specific known-bad-category prompt)
#     #####################################################

#     specific_boxes, s_boxes_raw, s_logits, s_phrases = run_grounding(
#         gd_image_tensor,
#         GROUNDING_PROMPT_SPECIFIC,
#         SPECIFIC_BOX_THRESHOLD,
#         SPECIFIC_TEXT_THRESHOLD,
#         W, H
#     )

#     print(f"  Specific-prompt objects (raw): {len(specific_boxes)}")

#     #####################################################
#     # STEP 2b — GroundingDINO (generic catch-all prompt)
#     #####################################################

#     generic_boxes, g_boxes_raw, g_logits, g_phrases = run_grounding(
#         gd_image_tensor,
#         GROUNDING_PROMPT_GENERIC,
#         GENERIC_BOX_THRESHOLD,
#         GENERIC_TEXT_THRESHOLD,
#         W, H
#     )

#     print(f"  Generic-prompt objects (raw): {len(generic_boxes)}")

#     # Raw GroundingDINO debug annotations (optional, one per prompt)
#     specific_debug = annotate(
#         image_source=gd_image_source,
#         boxes=s_boxes_raw,
#         logits=s_logits,
#         phrases=s_phrases
#     )
#     cv2.imwrite(os.path.join(out_dir, f"{stem}_groundingdino_specific.jpg"), specific_debug)

#     generic_debug = annotate(
#         image_source=gd_image_source,
#         boxes=g_boxes_raw,
#         logits=g_logits,
#         phrases=g_phrases
#     )
#     cv2.imwrite(os.path.join(out_dir, f"{stem}_groundingdino_generic.jpg"), generic_debug)

#     #####################################################
#     # STEP 2c — FIX: drop signage-strip false positives
#     #
#     # Applied to both prompts before anything else touches them, so a
#     # "lunch box" / "tiffin box" / "box" hit on the printed branding
#     # strip never makes it into the merge/exclusion pipeline at all.
#     #####################################################

#     def strip_degenerate_boxes(boxes_with_phrase):
#         kept = []
#         dropped_signage = 0
#         dropped_fullframe = 0
#         for gbox, phrase in boxes_with_phrase:
#             if looks_like_signage_strip(gbox, W, H):
#                 dropped_signage += 1
#                 continue
#             if is_degenerate_fullframe_box(gbox, W, H):
#                 dropped_fullframe += 1
#                 continue
#             kept.append((gbox, phrase))
#         if dropped_signage:
#             print(f"    Dropped {dropped_signage} signage-strip false positive(s)")
#         if dropped_fullframe:
#             print(f"    Dropped {dropped_fullframe} degenerate full-frame box(es)")
#         return kept

#     specific_boxes = strip_degenerate_boxes(specific_boxes)
#     generic_boxes = strip_degenerate_boxes(generic_boxes)

#     #####################################################
#     # STEP 3 — Merge + de-duplicate the two GroundingDINO passes
#     #
#     # Keep every specific-prompt box as-is (it has a useful label).
#     # Add generic-prompt boxes ONLY if they don't already overlap a
#     # specific-prompt box (otherwise they're the same physical object,
#     # just re-detected under a vague label like "object").
#     #####################################################

#     specific_only_boxes = [b for b, _ in specific_boxes]

#     merged_boxes = list(specific_boxes)  # (box, phrase)

#     for gbox, gphrase in generic_boxes:
#         if not overlaps_any(
#             gbox,
#             specific_only_boxes,
#             DEDUPE_IOU_THRESH,
#             DEDUPE_CONTAINMENT_THRESH
#         ):
#             merged_boxes.append((gbox, gphrase if gphrase else GENERIC_IMPURE_LABEL))

#     print(f"  Merged (deduped) objects: {len(merged_boxes)}")

#     #####################################################
#     # STEP 4 — Unknown objects (no overlap with YOLO beverages)
#     #
#     # FIX: use the looser BEVERAGE_* thresholds here instead of the
#     # overlaps_any() defaults, since GD's box on a bottle is often
#     # drawn much larger/looser than YOLO's tight beverage box.
#     #####################################################

#     unknown = []

#     for gbox, phrase in merged_boxes:

#         if not overlaps_any(
#             gbox,
#             beverage_boxes,
#             BEVERAGE_IOU_THRESH,
#             BEVERAGE_CONTAINMENT_THRESH
#         ):
#             unknown.append((gbox, phrase))

#     print(f"  Unknown (non-beverage) objects: {len(unknown)}")

#     #####################################################
#     # STEP 5 — Any detected non-beverage item -> impure
#     #####################################################

#     impure = False

#     for i, (box, phrase) in enumerate(unknown):

#         x1, y1, x2, y2 = box

#         x1 = max(0, x1)
#         y1 = max(0, y1)
#         x2 = min(W, x2)
#         y2 = min(H, y2)

#         if x2 <= x1 or y2 <= y1:
#             continue

#         clipped_box = (x1, y1, x2, y2)

#         item_name = phrase.strip() if phrase and phrase.strip() else IMPURE_LABEL

#         # If a KNOWN-BRAND cap (coca cola, sprite, fanta, kinley, etc.)
#         # sits inside this box, treat it as a beverage false-positive
#         # and skip it — regardless of whether an "other" cap is also
#         # present. Only known-brand caps can rescue a box.
#         if contains_known_cap(clipped_box, known_cap_boxes, CAP_CONTAINMENT_THRESH):
#             print(f"    Unknown {i+1}: {item_name} -> RESCUED (known-brand cap found inside) -> ignored")
#             continue

#         # FIX: vague "container"/"box" style labels on a near-uniform,
#         # low-contrast patch (dark shelf corner, compressor housing,
#         # reflection) are almost never a real object — drop them.
#         lowered_name = item_name.lower()
#         if any(term in lowered_name for term in VAGUE_CONTAINER_TERMS):
#             if is_low_contrast_region(image, clipped_box):
#                 print(f"    Unknown {i+1}: {item_name} -> DROPPED (low-contrast/no-object region) -> ignored")
#                 continue

#         impure = True

#         print(f"    Unknown {i+1}: {item_name} -> IMPURE")

#         draw_label_box(vis, (x1, y1, x2, y2), item_name, (0, 0, 255))

#     #####################################################
#     # STEP 6 — Final banner + save
#     #####################################################

#     draw_banner(vis, impure)

#     out_path = os.path.join(out_dir, f"{stem}_purity.jpg")
#     cv2.imwrite(out_path, vis)

#     print(f"  RESULT: {'IMPURE' if impure else 'PURE'}  ->  {out_path}")

#     return impure


# #########################################################
# # MAIN — RUN OVER A FOLDER OF IMAGES
# #########################################################

# if __name__ == "__main__":

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     image_paths = sorted(
#         p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
#         if p.lower().endswith(IMAGE_EXTENSIONS)
#     )

#     if not image_paths:
#         raise FileNotFoundError(
#             f"No images found in '{INPUT_DIR}' "
#             f"(looked for extensions: {IMAGE_EXTENSIONS})"
#         )

#     print(f"\nFound {len(image_paths)} image(s) in '{INPUT_DIR}'\n")

#     results = {}

#     for path in image_paths:
#         print(f"Processing: {path}")
#         impure = process_image(path, OUTPUT_DIR)

#         if impure is not None:
#             results[path] = "IMPURE" if impure else "PURE"

#         print()

#     #####################################################
#     # SUMMARY
#     #####################################################

#     print("=" * 50)
#     print("SUMMARY")
#     print("=" * 50)

#     pure_count = sum(1 for v in results.values() if v == "PURE")
#     impure_count = sum(1 for v in results.values() if v == "IMPURE")

#     for path, verdict in results.items():
#         print(f"{os.path.basename(path):40s} {verdict}")

#     print("-" * 50)
#     print(f"Total: {len(results)}   PURE: {pure_count}   IMPURE: {impure_count}")
#     print(f"\nAll annotated images saved in: {OUTPUT_DIR}/")


import os
import glob
import cv2
import torch
from ultralytics import YOLO

from groundingdino.util.inference import (
    load_model,
    predict,
    load_image,
    annotate
)

#########################################################
# CONFIG
#########################################################

INPUT_DIR = "downloaded_images"        # folder containing images to test
OUTPUT_DIR = "output-5"                  # folder where results will be saved
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

YOLO_MODEL = "data/availability_may_5th.pt"

# Bottle-cap model. Detects caps and classifies them by brand (coca cola,
# sprite, fanta, kinley, etc.) plus a catch-all "other(s)" class for caps
# that don't match a known beverage brand.
CAP_MODEL = "data/capmodel_june_26th.pt"

GROUNDING_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#########################################################
# GROUNDING PROMPTS
#
# Previously this was one 40-phrase caption (plus a fully generic
# "object . item . thing . stuff" catch-all). Two problems with that:
#
#   1. Bare nouns like "box" / "container" have almost no visual
#      signature -> they fire on shelf edges, compressor housing,
#      printed signage, shadows. Most of the earlier "FIX" filters
#      (signage-strip, full-frame, low-contrast) existed purely to
#      clean up noise from those two words.
#   2. One long concatenated caption increases cross-attention drift
#      between unrelated phrases, which is part of why box_threshold
#      had to be pushed as high as 0.75 to stay usable.
#
# Fix: split into short, thematically-grouped captions (shorter caption
# -> less cross-talk -> can run at a lower, saner box_threshold), and
# drop the bare "box"/"container" terms in favor of concrete objects.
#########################################################

PROMPT_TABLEWARE = (
    "cup . mug . drinking glass . bowl . plate . spoon . fork . knife . "
    "steel plate . steel bowl . steel glass"
)

PROMPT_FOOD = (
    "food packet . snack packet . chips packet . bread loaf . cake . "
    "egg . milk packet . curd cup . yogurt cup . "
    "paneer packet . ice cream tub . plastic food container . glass jar . "
    "steel tiffin box . lunch box . biscuit packet . chocolate bar . "
    "detergent packet . soap bar . shampoo bottle"
)

PROMPT_PERSONAL = (
    "mobile phone . wallet . keys . medicine bottle . cosmetic bottle . "
    "cloth . tissue paper . toy . umbrella . slipper . "
    "shoe . lighter . matchbox . battery"
)

# Produce / bagged-items — split out from PROMPT_FOOD and PROMPT_PERSONAL
# and given its own lower threshold below. A fruit seen through a
# translucent plastic bag is a much weaker visual match than a bare
# fruit on a shelf, so it needs a lower bar to be caught at all — but
# lowering the *whole* food/personal prompt would reopen the exact
# false-positive problem (GD misfiring "medicine bottle" on a real
# beverage bottle) that we just raised thresholds to fix. Isolating it
# lets this one category trade precision for recall without touching
# the others.
PROMPT_PRODUCE = "fruit . vegetable . plastic bag . fruit bag"

# Per-theme thresholds. Falls back to SPECIFIC_BOX_THRESHOLD /
# SPECIFIC_TEXT_THRESHOLD (defined below) when a theme has no override.
SPECIFIC_PROMPTS = {
    "tableware": PROMPT_TABLEWARE,
    "food": PROMPT_FOOD,
    "personal": PROMPT_PERSONAL,
    "produce": PROMPT_PRODUCE,
}

SPECIFIC_THEME_THRESHOLD_OVERRIDES = {
    # theme_name: (box_threshold, text_threshold)
    "produce": (0.28, 0.28),
}

# Fully generic catch-all. Kept ONLY as a secondary, non-authoritative
# signal — it is logged and drawn on its own debug image, but it does
# NOT by itself flip a shelf to IMPURE (see ENABLE_GENERIC_AS_IMPURE).
# Generic single-word grounding is inherently noisy in this model
# family; treat it as "worth a human glance", not ground truth.
GROUNDING_PROMPT_GENERIC = "object . item . thing . stuff"
ENABLE_GENERIC_AS_IMPURE = False

# Default specific-prompt thresholds (used by any theme without an
# entry in SPECIFIC_THEME_THRESHOLD_OVERRIDES above).
SPECIFIC_BOX_THRESHOLD = 0.42
SPECIFIC_TEXT_THRESHOLD = 0.42

GENERIC_BOX_THRESHOLD = 0.47
GENERIC_TEXT_THRESHOLD = 0.42

# YOLO confidence thresholds for the cap and SKU (availability) models.
#
# EXPERIMENT: raised from the production values (cap=0.20, sku=0.12)
# to test whether the squeeze/sauce-bottle misclassification (both
# models confidently calling a red-capped condiment bottle a real
# Fanta/Sprite SKU) clears up at a stricter bar. If it doesn't clear
# even here, that's a signal the models were never trained on
# condiment bottles as negatives, and no threshold will fix it — it
# needs those images added as hard negatives on the next retrain.
# Revert to 0.20 / 0.12 if this threshold change costs too much real
# beverage recall on your test set.
CAP_CONF_THRESHOLD = 0.35
SKU_CONF_THRESHOLD = 0.50



# Allowed: beverage bottles / soft drink bottles (glass, plastic, etc. —
# whatever the YOLO beverage model detects). Everything else is impure.
BEVERAGE_LABEL = "Beverage Bottle"
IMPURE_LABEL = "Impure Object"
GENERIC_IMPURE_LABEL = "Unidentified Item"

#########################################################
# HARD OVERRIDE — flagged cap / SKU classes
#
# If the CAP model or the availability (beverage) YOLO model returns a
# class name starting with "other" / "others", or containing
# "alcohol", the image is marked IMPURE immediately — no rescue, no
# GroundingDINO cross-check needed. This is a direct model-label
# signal (unrecognized brand cap, or a non-Coca-Cola / alcoholic SKU),
# so it overrides everything else in the pipeline.
#########################################################

FLAGGED_LABEL_PREFIXES = ("other", "others")
FLAGGED_LABEL_SUBSTRINGS = ("alcohol",)

FLAGGED_SKU_LABEL = "Flagged SKU"
FLAGGED_CAP_LABEL = "Flagged Cap"


def is_flagged_class_name(name):
    """
    True if a model class name should force the image to IMPURE:
    starts with "other"/"others", or contains "alcohol" anywhere
    (covers labels like "alcohol", "alcohol_beer", "other_alcohol").
    """
    n = name.strip().lower()
    if any(n.startswith(p) for p in FLAGGED_LABEL_PREFIXES):
        return True
    if any(s in n for s in FLAGGED_LABEL_SUBSTRINGS):
        return True
    return False

# If a generic-prompt box overlaps a specific-prompt box by more than
# this, treat them as the same physical object and drop the generic
# (less informative) one, keeping the specific label instead.
DEDUPE_IOU_THRESH = 0.5
DEDUPE_CONTAINMENT_THRESH = 0.6

# A GroundingDINO detection is dropped (treated as "actually just a
# beverage cap, not impure") if a KNOWN-BRAND cap (coca cola, sprite,
# fanta, kinley, etc.) is found mostly inside it. Flagged caps (class
# name starting with "other"/"others", or containing "alcohol") never
# rescue anything — and separately, their mere presence now forces the
# whole image to IMPURE via the hard override above, regardless of GD.
#
# Lowered from 0.5 -> 0.35: GroundingDINO regularly draws its box
# taller than the actual bottle (extending above the cap or below the
# shelf), so a tight cap box sitting only in the top slice of a tall GD
# box was never clearing 0.5 containment even on an obvious match.
CAP_CONTAINMENT_THRESH = 0.35
CAP_LABEL_KNOWN = "Brand Cap"

# Beverage-box exclusion thresholds — a bottle's real YOLO box is
# tight, but GD's box on the same bottle is often looser/oversized, so
# these are kept lower than a plain 0.5/0.6 default.
BEVERAGE_CONTAINMENT_THRESH = 0.4
BEVERAGE_IOU_THRESH = 0.5

#########################################################
# SAFETY-NET FILTERS
#
# Kept as a backstop even after cleaning up the prompts — real shelf
# photos still occasionally produce a stray signage-strip or
# degenerate box on a legitimate object phrase (e.g. "lunch box" on a
# printed branding strip), so these stay cheap and conservative.
#########################################################

# Signage / branding-strip filter — flat printed Coca-Cola / Sprite /
# Fanta branding at the top or bottom of a shelf is not a physical
# object. It's either a wide, short horizontal band (unoccluded) or a
# narrow sliver of one (partially occluded by bottles).
SIGNAGE_MIN_WIDTH_FRAC = 0.55
SIGNAGE_MAX_HEIGHT_FRAC = 0.18
SIGNAGE_MIN_ASPECT_RATIO = 4.0
SIGNAGE_MIN_ABS_WIDTH_FRAC = 0.12


def looks_like_signage_strip(box, W, H):
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return False

    near_top = y1 <= 0.12 * H
    near_bottom = y2 >= 0.88 * H
    near_edge = near_top or near_bottom

    wide_enough = box_w >= SIGNAGE_MIN_WIDTH_FRAC * W
    short_enough = box_h <= SIGNAGE_MAX_HEIGHT_FRAC * H
    if wide_enough and short_enough and near_edge:
        return True

    aspect_ratio = box_w / box_h
    has_min_width = box_w >= SIGNAGE_MIN_ABS_WIDTH_FRAC * W
    if aspect_ratio >= SIGNAGE_MIN_ASPECT_RATIO and has_min_width and near_edge:
        return True

    return False


# Degenerate full-frame box filter — occasionally GD returns a box
# covering nearly the whole image for a weakly-matched phrase. Not a
# real localized object.
MAX_BOX_AREA_FRAC = 0.65


def is_degenerate_fullframe_box(box, W, H):
    x1, y1, x2, y2 = box
    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    box_area = box_w * box_h
    frame_area = W * H

    if frame_area <= 0:
        return False

    return (box_area / frame_area) > MAX_BOX_AREA_FRAC


# Low-contrast / low-variance region filter — vague container-ish
# labels occasionally fire on a dark, near-uniform patch (empty shelf
# corner, compressor housing, shadow) with no real object present.
VAGUE_CONTAINER_TERMS = ("container", "tiffin box", "lunch box", "jar")
MIN_REGION_STD = 12.0


def is_low_contrast_region(image, box):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return True

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return True

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < MIN_REGION_STD


#########################################################
# Low-light preprocessing for the beverage detector
#
# Some real bottles are missed by the beverage YOLO model entirely
# (not mislabeled — never detected) because they sit in dark /
# underexposed regions (shelf edges, gaps between rows, poor
# lighting). CLAHE boosts local contrast in dark regions without
# blowing out already-bright areas, helping recover those bottles.
#
# Applied only to the copy fed to the beverage YOLO model — the
# original `image` used for GroundingDINO, cap detection, and the
# final saved visualization is untouched.
#########################################################

def enhance_for_dark_regions(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


#########################################################
# DRAWING HELPERS
#########################################################

def draw_label_box(img, box, text, color, font_scale=0.9, thickness=2):
    """Draw a bounding box with a filled label background and readable text."""
    x1, y1, x2, y2 = box

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )

    text_y = max(th + 8, y1)

    cv2.rectangle(
        img,
        (x1, text_y - th - baseline - 4),
        (x1 + tw + 8, text_y + baseline),
        color,
        -1
    )

    cv2.putText(
        img,
        text,
        (x1 + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_banner(img, impure):
    """Draw the PURE / IMPURE result banner at the top-left of the image."""
    if impure:
        banner = "RESULT : IMPURE"
        banner_color = (0, 0, 255)
    else:
        banner = "RESULT : PURE"
        banner_color = (0, 180, 0)

    cv2.rectangle(img, (0, 0), (520, 55), banner_color, -1)

    cv2.putText(
        img,
        banner,
        (15, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3
    )


#########################################################
# IOU / CONTAINMENT
#########################################################

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter)


def containment_ratio(boxA, boxB):
    """
    Fraction of the SMALLER box that lies inside the other box.
    Unlike IoU, this isn't penalized by a big size mismatch — so a
    tiny GroundingDINO box drawn on a bottle cap/label/reflection
    that sits inside a much larger YOLO beverage box still scores
    close to 1.0 here, even though its IoU would be near 0.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    smaller_area = min(areaA, areaB)
    if smaller_area <= 0:
        return 0

    return inter / smaller_area


def overlaps_any(candidate_box, other_boxes, iou_thresh=0.5, containment_thresh=0.6):
    """
    True if candidate_box substantially overlaps ANY box in other_boxes,
    either via IoU or containment (smaller box mostly inside larger box).
    """
    for bbox in other_boxes:
        if iou(candidate_box, bbox) > iou_thresh:
            return True
        if containment_ratio(candidate_box, bbox) > containment_thresh:
            return True
    return False


def contains_known_cap(candidate_box, known_cap_boxes, containment_thresh=0.5):
    """
    True if any KNOWN-BRAND cap box is mostly contained inside
    candidate_box (i.e. the cap sits inside the GroundingDINO-detected
    object). Caps are small, so we check containment (fraction of the
    cap's own area that lies inside candidate_box) rather than IoU,
    which would unfairly penalize the large size mismatch.
    """
    for cap_box in known_cap_boxes:
        if containment_ratio(cap_box, candidate_box) > containment_thresh:
            return True
    return False


#########################################################
# LOAD MODELS
#########################################################

print("Loading Beverage YOLO...")
beverage_model = YOLO(YOLO_MODEL)

print("Loading Cap YOLO...")
cap_model = YOLO(CAP_MODEL)

print("Loading GroundingDINO...")
grounding_model = load_model(
    GROUNDING_CONFIG,
    GROUNDING_WEIGHTS
)

#########################################################
# HELPER — run GroundingDINO + convert to absolute boxes
#########################################################

def run_grounding(image_tensor, prompt, box_thresh, text_thresh, W, H):
    """
    Runs GroundingDINO with the given prompt/thresholds and returns a
    list of (abs_box, phrase) tuples, where abs_box is [x1, y1, x2, y2]
    in absolute pixel coordinates.
    """
    boxes, logits, phrases = predict(
        model=grounding_model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        device=DEVICE
    )

    abs_boxes = []

    for b, phrase in zip(boxes, phrases):
        cx, cy, w, h = b.cpu().numpy()

        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        abs_boxes.append(([x1, y1, x2, y2], phrase))

    return abs_boxes, boxes, logits, phrases


def strip_degenerate_boxes(boxes_with_phrase, W, H, label=""):
    """Drop signage-strip and full-frame degenerate detections."""
    kept = []
    dropped_signage = 0
    dropped_fullframe = 0

    for gbox, phrase in boxes_with_phrase:
        if looks_like_signage_strip(gbox, W, H):
            dropped_signage += 1
            continue
        if is_degenerate_fullframe_box(gbox, W, H):
            dropped_fullframe += 1
            continue
        kept.append((gbox, phrase))

    if dropped_signage:
        print(f"    [{label}] Dropped {dropped_signage} signage-strip false positive(s)")
    if dropped_fullframe:
        print(f"    [{label}] Dropped {dropped_fullframe} degenerate full-frame box(es)")

    return kept


#########################################################
# PER-IMAGE PIPELINE
#########################################################

def process_image(image_path, out_dir):
    """
    Runs the full YOLO + GroundingDINO(split thematic prompts) + purity
    pipeline on a single image and saves the annotated visualization +
    raw GroundingDINO debug annotations into out_dir.

    Returns True if the image is classified IMPURE, False if PURE.
    """

    stem = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)

    if image is None:
        print(f"  [SKIP] Could not read image: {image_path}")
        return None

    gd_image_source, gd_image_tensor = load_image(image_path)

    H, W = image.shape[:2]

    # This will accumulate all drawing (YOLO boxes, GD boxes, banner)
    vis = image.copy()

    #####################################################
    # STEP 1 — Beverage Detection (YOLO)
    #
    # Runs on a CLAHE-enhanced copy so dark/underexposed bottles
    # (shelf edges, shadowed gaps) are more likely to be detected.
    # `image` itself stays untouched for GD, cap detection, and the
    # final saved visualization.
    #####################################################

    beverage_input = enhance_for_dark_regions(image)
    yolo_result = beverage_model(beverage_input, verbose=False, conf=SKU_CONF_THRESHOLD)[0]

    beverage_boxes = []
    flagged_sku_hits = []  # (box, cls_name) for "other"/"others"/alcohol SKUs

    for box in yolo_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = (
            beverage_model.names.get(cls_id, str(cls_id))
            if isinstance(beverage_model.names, dict)
            else beverage_model.names[cls_id]
        )

        if is_flagged_class_name(cls_name):
            flagged_sku_hits.append(([x1, y1, x2, y2], cls_name))
            color = (0, 0, 255)  # red — flagged SKU, forces IMPURE
            label = f"{FLAGGED_SKU_LABEL}: {cls_name}"
        else:
            beverage_boxes.append([x1, y1, x2, y2])
            color = (0, 255, 0)  # green — normal beverage
            label = BEVERAGE_LABEL

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis, label, (x1, max(25, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    print(f"  Detected beverages: {len(beverage_boxes)}  |  flagged SKUs: {len(flagged_sku_hits)}")
    for _, cls_name in flagged_sku_hits:
        print(f"    Flagged SKU class: '{cls_name}' -> forces IMPURE")

    #####################################################
    # STEP 1b — Bottle Cap Detection (YOLO)
    #
    # Known-brand caps (coca cola, sprite, fanta, kinley, ...) are used
    # later to rescue GroundingDINO boxes that are actually just a
    # branded bottle cap (false-positive impurity). Caps classified as
    # "other(s)" (unrecognized brand) are tracked separately and are
    # NOT allowed to rescue a box — an unrecognized cap keeps the
    # object flagged as impure.
    #####################################################

    cap_result = cap_model(image, verbose=False, conf=CAP_CONF_THRESHOLD)[0]

    known_cap_boxes = []
    other_cap_boxes = []
    flagged_cap_hits = []  # (box, cls_name) for "other"/"others"/alcohol caps

    for box in cap_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        cls_name = cap_model.names.get(cls_id, str(cls_id)) if isinstance(cap_model.names, dict) else cap_model.names[cls_id]

        if is_flagged_class_name(cls_name):
            other_cap_boxes.append([x1, y1, x2, y2])
            flagged_cap_hits.append(([x1, y1, x2, y2], cls_name))
            color = (0, 0, 255)   # red — flagged cap, forces IMPURE
            label = f"{FLAGGED_CAP_LABEL}: {cls_name}"
        else:
            known_cap_boxes.append([x1, y1, x2, y2])
            color = (0, 255, 255)   # yellow — known brand cap
            label = f"{CAP_LABEL_KNOWN}: {cls_name}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis, label, (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    print(f"  Detected caps: {len(known_cap_boxes)} known-brand, {len(other_cap_boxes)} flagged")
    for _, cls_name in flagged_cap_hits:
        print(f"    Flagged cap class: '{cls_name}' -> forces IMPURE")

    #####################################################
    # STEP 2 — GroundingDINO, split thematic prompts
    #
    # Each theme runs as its own short caption instead of one long
    # concatenated string. Shorter captions -> less cross-attention
    # drift between unrelated phrases -> can run at a normal
    # box_threshold instead of needing it pushed to 0.75.
    #####################################################

    specific_boxes = []

    for theme_name, theme_prompt in SPECIFIC_PROMPTS.items():
        box_thresh, text_thresh = SPECIFIC_THEME_THRESHOLD_OVERRIDES.get(
            theme_name, (SPECIFIC_BOX_THRESHOLD, SPECIFIC_TEXT_THRESHOLD)
        )

        boxes_abs, boxes_raw, logits, phrases = run_grounding(
            gd_image_tensor,
            theme_prompt,
            box_thresh,
            text_thresh,
            W, H
        )

        print(f"  [{theme_name}] objects (raw, box_thresh={box_thresh}): {len(boxes_abs)}")

        debug_frame = annotate(
            image_source=gd_image_source,
            boxes=boxes_raw,
            logits=logits,
            phrases=phrases
        )
        cv2.imwrite(os.path.join(out_dir, f"{stem}_groundingdino_{theme_name}.jpg"), debug_frame)

        boxes_abs = strip_degenerate_boxes(boxes_abs, W, H, label=theme_name)
        specific_boxes.extend(boxes_abs)

    print(f"  Specific-prompt objects (after filtering, all themes): {len(specific_boxes)}")

    #####################################################
    # STEP 2b — GroundingDINO generic catch-all (secondary signal only)
    #
    # Logged and drawn to its own debug image, but does NOT flip a
    # shelf to IMPURE unless ENABLE_GENERIC_AS_IMPURE is turned on.
    # Generic single-word grounding is inherently noisy; treat it as
    # "worth a human glance" rather than ground truth.
    #####################################################

    generic_boxes, g_boxes_raw, g_logits, g_phrases = run_grounding(
        gd_image_tensor,
        GROUNDING_PROMPT_GENERIC,
        GENERIC_BOX_THRESHOLD,
        GENERIC_TEXT_THRESHOLD,
        W, H
    )

    print(f"  Generic-prompt objects (raw): {len(generic_boxes)}")

    generic_debug = annotate(
        image_source=gd_image_source,
        boxes=g_boxes_raw,
        logits=g_logits,
        phrases=g_phrases
    )
    cv2.imwrite(os.path.join(out_dir, f"{stem}_groundingdino_generic.jpg"), generic_debug)

    generic_boxes = strip_degenerate_boxes(generic_boxes, W, H, label="generic")

    #####################################################
    # STEP 3 — Merge specific themes + (optionally) generic pass
    #
    # Generic boxes are only added if ENABLE_GENERIC_AS_IMPURE is on,
    # and even then only if they don't already overlap a specific box
    # (otherwise it's the same physical object under a vaguer label).
    #####################################################

    specific_only_boxes = [b for b, _ in specific_boxes]

    merged_boxes = list(specific_boxes)  # (box, phrase)

    if ENABLE_GENERIC_AS_IMPURE:
        for gbox, gphrase in generic_boxes:
            if not overlaps_any(
                gbox,
                specific_only_boxes,
                DEDUPE_IOU_THRESH,
                DEDUPE_CONTAINMENT_THRESH
            ):
                merged_boxes.append((gbox, gphrase if gphrase else GENERIC_IMPURE_LABEL))

    print(f"  Merged (deduped) objects considered for impurity: {len(merged_boxes)}")

    #####################################################
    # STEP 4 — Unknown objects (no overlap with YOLO beverages)
    #
    # Looser BEVERAGE_* thresholds are used here instead of the
    # overlaps_any() defaults, since GD's box on a bottle is often
    # drawn much larger/looser than YOLO's tight beverage box.
    #####################################################

    unknown = []

    for gbox, phrase in merged_boxes:
        if not overlaps_any(
            gbox,
            beverage_boxes,
            BEVERAGE_IOU_THRESH,
            BEVERAGE_CONTAINMENT_THRESH
        ):
            unknown.append((gbox, phrase))

    print(f"  Unknown (non-beverage) objects: {len(unknown)}")

    #####################################################
    # STEP 5 — Any detected non-beverage item -> impure
    #####################################################

    impure = False

    for i, (box, phrase) in enumerate(unknown):
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        clipped_box = (x1, y1, x2, y2)
        item_name = phrase.strip() if phrase and phrase.strip() else IMPURE_LABEL

        # If a KNOWN-BRAND cap (coca cola, sprite, fanta, kinley, etc.)
        # sits inside this box, treat it as a beverage false-positive
        # and skip it — regardless of whether an "other" cap is also
        # present. Only known-brand caps can rescue a box.
        if contains_known_cap(clipped_box, known_cap_boxes, CAP_CONTAINMENT_THRESH):
            print(f"    Unknown {i+1}: {item_name} -> RESCUED (known-brand cap found inside) -> ignored")
            continue

        # Vague container-ish labels on a near-uniform, low-contrast
        # patch (dark shelf corner, compressor housing, reflection)
        # are almost never a real object — drop them.
        lowered_name = item_name.lower()
        if any(term in lowered_name for term in VAGUE_CONTAINER_TERMS):
            if is_low_contrast_region(image, clipped_box):
                print(f"    Unknown {i+1}: {item_name} -> DROPPED (low-contrast/no-object region) -> ignored")
                continue

        impure = True
        print(f"    Unknown {i+1}: {item_name} -> IMPURE")
        draw_label_box(vis, (x1, y1, x2, y2), item_name, (0, 0, 255))

    #####################################################
    # STEP 5b — HARD OVERRIDE: flagged cap / SKU classes
    #
    # Any "other"/"others"/alcohol cap or SKU detected in Steps 1/1b
    # forces IMPURE outright, independent of whatever GroundingDINO
    # did or didn't find. Boxes were already drawn in red during
    # detection, so no extra drawing is needed here — just the verdict
    # and a print explaining why.
    #####################################################

    if flagged_sku_hits or flagged_cap_hits:
        if not impure:
            print("    Forcing IMPURE due to flagged cap/SKU class(es):")
        for _, cls_name in flagged_sku_hits:
            print(f"      - SKU class '{cls_name}'")
        for _, cls_name in flagged_cap_hits:
            print(f"      - Cap class '{cls_name}'")
        impure = True

    #####################################################
    # STEP 6 — Final banner + save
    #####################################################

    draw_banner(vis, impure)

    out_path = os.path.join(out_dir, f"{stem}_purity.jpg")
    cv2.imwrite(out_path, vis)

    print(f"  RESULT: {'IMPURE' if impure else 'PURE'}  ->  {out_path}")

    return impure


#########################################################
# MAIN — RUN OVER A FOLDER OF IMAGES
#########################################################

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = sorted(
        p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
        if p.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in '{INPUT_DIR}' "
            f"(looked for extensions: {IMAGE_EXTENSIONS})"
        )

    print(f"\nFound {len(image_paths)} image(s) in '{INPUT_DIR}'\n")

    results = {}

    for path in image_paths:
        print(f"Processing: {path}")
        impure = process_image(path, OUTPUT_DIR)

        if impure is not None:
            results[path] = "IMPURE" if impure else "PURE"

        print()

    #####################################################
    # SUMMARY
    #####################################################

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)

    pure_count = sum(1 for v in results.values() if v == "PURE")
    impure_count = sum(1 for v in results.values() if v == "IMPURE")

    for path, verdict in results.items():
        print(f"{os.path.basename(path):40s} {verdict}")

    print("-" * 50)
    print(f"Total: {len(results)}   PURE: {pure_count}   IMPURE: {impure_count}")
    print(f"\nAll annotated images saved in: {OUTPUT_DIR}/")