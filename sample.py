# # from ultralytics import YOLO
# # model = YOLO("data/visicooler.pt")
# # print(model.names)  

# # # from ultralytics import YOLO

# # # model1 = YOLO("data/latest_activation.pt")
# # # model2 = YOLO("data/activation_april.pt")

# # # print(model1.names == model2.names)


# # # from ultralytics import YOLO

# # # model1 = YOLO("data/latest_activation.pt")
# # # model2 = YOLO("data/activation_april.pt")

# # # print("Model 1 Classes:")
# # # print(model1.names)

# # # print("\nModel 2 Classes:")
# # # print(model2.names)

# # # print("\nAre both identical?", model1.names == model2.names)

# # # print("\nDifferences:")

# # # all_keys = sorted(set(model1.names.keys()) | set(model2.names.keys()))

# # # difference_found = False

# # # for key in all_keys:
# # #     name1 = model1.names.get(key, "<Not Present>")
# # #     name2 = model2.names.get(key, "<Not Present>")

# # #     if name1 != name2:
# # #         difference_found = True
# # #         print(f"Class ID {key}:")
# # #         print(f"  Model 1 -> {name1}")
# # #         print(f"  Model 2 -> {name2}")
# # #         print()

# # # if not difference_found:
# # #     print("No differences found. Both models have identical class mappings.")

# # # from ultralytics import YOLO
# # # from collections import defaultdict
# # # import cv2

# # # model = YOLO("data/capmodel_june_26th.pt")

# # # results = model(
# # #     "IMG-MR0LGS6K-GXGS.jpg",
# # #     conf=0.2,iou=0.5
# # # )

# # # class_counts = defaultdict(int)

# # # for r in results:
# # #     # Count classes
# # #     for box in r.boxes:
# # #         class_id = int(box.cls)
# # #         class_name = model.names[class_id]
# # #         class_counts[class_name] += 1

# # #     # Get annotated image
# # #     annotated_img = r.plot()   # draws boxes + labels

# # #     # Save annotated image
# # #     cv2.imwrite("annotated_output.png", annotated_img)

# # # # Print class-wise counts
# # # print("Class-wise counts:")
# # # for cls, count in class_counts.items():
# # #     print(f"{cls}: {count}")

# # # print("Total detections:", sum(class_counts.values()))



# # # from ultralytics import YOLO
# # # from collections import defaultdict
# # # import cv2
# # # import os
# # # import pandas as pd

# # # model = YOLO("data/activation_april.pt")

# # # image_paths = [
# # #     "image (9).jpg",
# # #     "image (10).jpg",
# # #     "image (11).jpg",
# # #     "image (13).jpg",
# # #     "image (14).jpg",
# # #     "image (15).jpg",
# # #     "image (16).jpg",
# # #     "image (17).jpg",
# # #     "image (18).jpg"
# # # ]

# # # results = model(image_paths, conf=0.01, iou=0.1)

# # # overall_counts = defaultdict(int)
# # # rows = []  # for CSV

# # # for idx, r in enumerate(results):
# # #     image_name = os.path.basename(image_paths[idx])
# # #     class_counts = defaultdict(int)

# # #     # Count SKUs
# # #     for box in r.boxes:
# # #         class_id = int(box.cls)
# # #         class_name = model.names[class_id]
# # #         class_counts[class_name] += 1
# # #         overall_counts[class_name] += 1

# # #     # Save annotated image
# # #     annotated_img = r.plot()
# # #     cv2.imwrite(f"annotated_{image_name}", annotated_img)

# # #     # Print per image
# # #     print(f"\nImage: {image_name}")
# # #     for cls, count in class_counts.items():
# # #         print(f"{cls}: {count}")

# # #         # Save row for CSV
# # #         rows.append({
# # #             "image_name": image_name,
# # #             "sku": cls,
# # #             "count": count
# # #         })

# # #     print("Total:", sum(class_counts.values()))

# # # # Save CSV
# # # df = pd.DataFrame(rows)
# # # df.to_csv("sku_counts_per_image.csv", index=False)

# # # # Overall summary
# # # print("\n=== Overall Counts ===")
# # # for cls, count in overall_counts.items():
# # #     print(f"{cls}: {count}")

# # # print("Total detections:", sum(overall_counts.values()))


# from ultralytics import YOLO
# from collections import defaultdict
# import cv2
# import os
# import glob
# import pandas as pd

# model = YOLO("data/visicooler.pt")

# # Folder containing images
# image_folder = "downloaded_images"

# # Folder to save annotated images
# output_folder = "annotated_images"
# os.makedirs(output_folder, exist_ok=True)

# # Read all image files from folder
# image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + \
#               glob.glob(os.path.join(image_folder, "*.png")) + \
#               glob.glob(os.path.join(image_folder, "*.jpeg"))

# results = model(image_paths, conf=0.1, iou=0.1)

# overall_counts = defaultdict(int)
# rows = []  # for CSV

# for idx, r in enumerate(results):
#     image_name = os.path.basename(image_paths[idx])
#     class_counts = defaultdict(int)

#     # Count SKUs
#     for box in r.boxes:
#         class_id = int(box.cls)
#         class_name = model.names[class_id]
#         class_counts[class_name] += 1
#         overall_counts[class_name] += 1

#     # Save annotated image
#     annotated_img = r.plot()

#     annotated_image_path = os.path.join(
#         output_folder,
#         f"annotated_{image_name}"
#     )

#     cv2.imwrite(annotated_image_path, annotated_img)

#     # Print per image
#     print(f"\nImage: {image_name}")
#     for cls, count in class_counts.items():
#         print(f"{cls}: {count}")

#         # Save row for CSV
#         rows.append({
#             "image_name": image_name,
#             "sku": cls,
#             "count": count
#         })

#     print("Total:", sum(class_counts.values()))

# # Save CSV
# df = pd.DataFrame(rows)
# df.to_csv("sku_counts_per_image.csv", index=False)

# # Overall summary
# print("\n=== Overall Counts ===")
# for cls, count in overall_counts.items():
#     print(f"{cls}: {count}")

# print("Total detections:", sum(overall_counts.values()))



# # import cv2
# # import torch
# # import numpy as np

# # from ultralytics import YOLO
# # from PIL import Image

# # from transformers import AutoProcessor
# # from transformers import AutoModel

# # from groundingdino.util.inference import load_model, predict, load_image

# # #########################################################
# # # CONFIG
# # #########################################################

# # IMAGE_PATH = "10.jpg"

# # YOLO_MODEL = "data/availability_may_5th.pt"

# # GROUNDING_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# # GROUNDING_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # #########################################################
# # # LOAD MODELS
# # #########################################################

# # print("Loading Beverage YOLO...")
# # beverage_model = YOLO(YOLO_MODEL)

# # print("Loading GroundingDINO...")
# # grounding_model = load_model(
# #     GROUNDING_CONFIG,
# #     GROUNDING_WEIGHTS
# # )

# # print("Loading SigLIP...")
# # processor = AutoProcessor.from_pretrained(
# #     "google/siglip-base-patch16-224"
# # )

# # siglip = AutoModel.from_pretrained(
# #     "google/siglip-base-patch16-224"
# # ).to(DEVICE)

# # #########################################################
# # # IMAGE
# # #########################################################

# # # Raw image for YOLO + cropping (BGR, via OpenCV)
# # image = cv2.imread(IMAGE_PATH)

# # if image is None:
# #     raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")

# # # GroundingDINO needs its own preprocessed tensor
# # gd_image_source, gd_image_tensor = load_image(IMAGE_PATH)

# # #########################################################
# # # STEP 1
# # # Beverage Detection
# # #########################################################

# # print("\nRunning Beverage YOLO...")

# # yolo_result = beverage_model(image)[0]

# # beverage_boxes = []

# # for box in yolo_result.boxes:
# #     x1, y1, x2, y2 = map(int, box.xyxy[0])
# #     beverage_boxes.append([x1, y1, x2, y2])

# # print("Detected beverages:", len(beverage_boxes))

# # #########################################################
# # # STEP 2
# # # Detect Everything
# # #########################################################

# # print("\nRunning GroundingDINO...")

# # boxes, logits, phrases = predict(
# #     model=grounding_model,
# #     image=gd_image_tensor,
# #     caption="all objects",
# #     box_threshold=0.30,
# #     text_threshold=0.25,
# #     device=DEVICE
# # )

# # print("Total objects:", len(boxes))

# # #########################################################
# # # STEP 3
# # # Convert boxes (relative cx,cy,w,h -> absolute x1,y1,x2,y2)
# # #########################################################

# # H, W = image.shape[:2]

# # gd_boxes = []

# # for b in boxes:
# #     cx, cy, w, h = b.numpy()

# #     x1 = int((cx - w / 2) * W)
# #     y1 = int((cy - h / 2) * H)
# #     x2 = int((cx + w / 2) * W)
# #     y2 = int((cy + h / 2) * H)

# #     gd_boxes.append([x1, y1, x2, y2])

# # #########################################################
# # # IOU
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

# # #########################################################
# # # STEP 4
# # # Find Unknown Objects (GroundingDINO boxes not matched to a beverage)
# # #########################################################

# # unknown = []

# # for gbox in gd_boxes:
# #     overlap = False

# #     for bbox in beverage_boxes:
# #         if iou(gbox, bbox) > 0.5:
# #             overlap = True
# #             break

# #     if not overlap:
# #         unknown.append(gbox)

# # print("Unknown Objects:", len(unknown))

# # #########################################################
# # # STEP 5
# # # Classify each unknown object with SigLIP (single pass)
# # #########################################################

# # labels = [
# #     "beverage bottle",
# #     "soft drink bottle",
# #     "beer bottle",
# #     "beverage can",
# #     "plastic bag",
# #     "plastic jar",
# #     "steel utensil",
# #     "steel glass",
# #     "cup",
# #     "food",
# #     "cleaning bottle"
# # ]

# # allowed = [
# #     "beverage bottle",
# #     "soft drink bottle"
# # ]

# # impure = False
# # predicted_labels = []

# # for i, box in enumerate(unknown):
# #     x1, y1, x2, y2 = box

# #     # Clip box to image bounds to avoid negative/out-of-range crops
# #     x1, y1 = max(0, x1), max(0, y1)
# #     x2, y2 = min(W, x2), min(H, y2)

# #     crop = image[y1:y2, x1:x2]

# #     if crop.size == 0:
# #         continue

# #     crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

# #     inputs = processor(
# #         text=labels,
# #         images=crop,
# #         return_tensors="pt",
# #         padding=True
# #     ).to(DEVICE)

# #     with torch.no_grad():
# #         outputs = siglip(**inputs)
# #         logits_per_image = outputs.logits_per_image
# #         probs = logits_per_image.softmax(dim=-1)
# #         idx = probs.argmax().item()

# #     label = labels[idx]
# #     predicted_labels.append(label)

# #     print(f"Unknown {i + 1}: {label}")

# #     if label not in allowed:
# #         impure = True

# # #########################################################
# # # FINAL
# # #########################################################

# # if impure:
# #     print("\nFINAL RESULT : IMPURE")
# # else:
# #     print("\nFINAL RESULT : PURE")



# # import os
# # import json
# # from PIL import Image
# # from google import genai

# # # --------------------------------------------------
# # # CONFIG
# # # --------------------------------------------------

# # GOOGLE_API_KEY = "AIzaSyDxKCmR1Q9-sTCvXx6h3kEsaqwJktuPC70" # Or replace with your key

# # client = genai.Client(api_key=GOOGLE_API_KEY)

# # MODEL = "gemini-2.5-flash"

# # PURITY_PROMPT = """
# # You are a visicooler purity inspector.

# # Classify the image as PURE or IMPURE.

# # PURE:
# # - Only approved PET soft drink bottles, approved glass soft drink bottles, and approved juice cartons/tetra packs are visible.

# # IMPURE:
# # - Any other object is visible, even partially.

# # The following are always IMPURE:
# # - Beer bottles
# # - Beverage cans
# # - Plastic items
# # - Steel utensils
# # - Curd/yogurt cups
# # - Plastic covers or bags
# # - Food items
# # - Household items
# # - Any object that is not an approved beverage product.

# # Inspect the entire image, including:
# # - Front
# # - Back
# # - Left
# # - Right
# # - Top
# # - Bottom
# # - Corners
# # - Shelves
# # - Partially visible or occluded objects.

# # If there is any uncertainty, respond with IMPURE.

# # Return exactly one word:

# # PURE

# # or

# # IMPURE
# # """

# # # --------------------------------------------------
# # # GEMINI
# # # --------------------------------------------------

# # def classify_image(image_path):

# #     image = Image.open(image_path)

# #     response = client.models.generate_content(
# #         model=MODEL,
# #         contents=[
# #             PURITY_PROMPT,
# #             image
# #         ]
# #     )

# #     return response.text.strip()


# # # --------------------------------------------------
# # # PROCESS SINGLE IMAGE
# # # --------------------------------------------------

# # def process_image(image_path):
# #     return classify_image(image_path)


# # # --------------------------------------------------
# # # PROCESS FOLDER
# # # --------------------------------------------------

# # def run_on_folder(folder_path):

# #     results = {}

# #     for fname in sorted(os.listdir(folder_path)):

# #         if fname.lower().endswith((".jpg", ".jpeg", ".png")):

# #             full_path = os.path.join(folder_path, fname)

# #             print(f"Processing {fname}...")

# #             try:

# #                 result = process_image(full_path)

# #                 print(" ->", result)

# #                 results[fname] = result

# #             except Exception as e:

# #                 print("Error:", e)

# #                 results[fname] = "ERROR"

# #     return results


# # # --------------------------------------------------
# # # MAIN
# # # --------------------------------------------------

# # if __name__ == "__main__":

# #     folder = r"shelf_images\shelf_images"

# #     output = run_on_folder(folder)

# #     with open("results.json", "w") as f:
# #         json.dump(output, f, indent=4)

# #     print("\nDone!")



# # import cv2
# # import torch
# # from ultralytics import YOLO
# # from PIL import Image

# # from transformers import AutoProcessor
# # from transformers import AutoModel

# # from groundingdino.util.inference import (
# #     load_model,
# #     predict,
# #     load_image,
# #     annotate
# # )

# # #########################################################
# # # CONFIG
# # #########################################################

# # IMAGE_PATH = "10.jpg"

# # YOLO_MODEL = "data/availability_may_5th.pt"

# # GROUNDING_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# # GROUNDING_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # #########################################################
# # # LOAD MODELS
# # #########################################################

# # print("Loading Beverage YOLO...")
# # beverage_model = YOLO(YOLO_MODEL)

# # print("Loading GroundingDINO...")
# # grounding_model = load_model(
# #     GROUNDING_CONFIG,
# #     GROUNDING_WEIGHTS
# # )

# # print("Loading SigLIP...")
# # processor = AutoProcessor.from_pretrained(
# #     "google/siglip-base-patch16-224"
# # )

# # siglip = AutoModel.from_pretrained(
# #     "google/siglip-base-patch16-224"
# # ).to(DEVICE)

# # #########################################################
# # # IMAGE
# # #########################################################

# # image = cv2.imread(IMAGE_PATH)

# # if image is None:
# #     raise FileNotFoundError(f"Could not read image at {IMAGE_PATH}")

# # gd_image_source, gd_image_tensor = load_image(IMAGE_PATH)

# # #########################################################
# # # STEP 1
# # # Beverage Detection
# # #########################################################

# # print("\nRunning Beverage YOLO...")

# # yolo_result = beverage_model(image)[0]

# # beverage_boxes = []

# # for box in yolo_result.boxes:
# #     x1, y1, x2, y2 = map(int, box.xyxy[0])
# #     beverage_boxes.append([x1, y1, x2, y2])

# # print("Detected beverages:", len(beverage_boxes))

# # #########################################################
# # # STEP 2
# # # GroundingDINO
# # #########################################################

# # print("\nRunning GroundingDINO...")

# # boxes, logits, phrases = predict(
# #     model=grounding_model,
# #     image=gd_image_tensor,
# #     caption="all objects",
# #     box_threshold=0.30,
# #     text_threshold=0.25,
# #     device=DEVICE
# # )

# # print("Total objects:", len(boxes))

# # #########################################################
# # # SAVE GROUNDINGDINO ANNOTATED IMAGE
# # #########################################################

# # annotated_frame = annotate(
# #     image_source=gd_image_source,
# #     boxes=boxes,
# #     logits=logits,
# #     phrases=phrases
# # )

# # cv2.imwrite("groundingdino_output.jpg", annotated_frame)

# # print("Saved GroundingDINO annotation -> groundingdino_output.jpg")

# # #########################################################
# # # STEP 3
# # # Convert Relative Boxes -> Absolute
# # #########################################################

# # H, W = image.shape[:2]

# # gd_boxes = []

# # for b in boxes:
# #     cx, cy, w, h = b.cpu().numpy()

# #     x1 = int((cx - w / 2) * W)
# #     y1 = int((cy - h / 2) * H)
# #     x2 = int((cx + w / 2) * W)
# #     y2 = int((cy + h / 2) * H)

# #     gd_boxes.append([x1, y1, x2, y2])

# # #########################################################
# # # IOU
# # #########################################################

# # def iou(boxA, boxB):

# #     xA = max(boxA[0], boxB[0])
# #     yA = max(boxA[1], boxB[1])

# #     xB = min(boxA[2], boxB[2])
# #     yB = min(boxA[3], boxB[3])

# #     inter = max(0, xB - xA) * max(0, yB - yA)

# #     if inter == 0:
# #         return 0

# #     areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
# #     areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

# #     return inter/(areaA+areaB-inter)

# # #########################################################
# # # STEP 4
# # # UNKNOWN OBJECTS
# # #########################################################

# # unknown = []

# # for gbox in gd_boxes:

# #     overlap = False

# #     for bbox in beverage_boxes:

# #         if iou(gbox, bbox) > 0.5:
# #             overlap = True
# #             break

# #     if not overlap:
# #         unknown.append(gbox)

# # print("Unknown Objects:", len(unknown))

# # #########################################################
# # # STEP 5
# # # SIGLIP CLASSIFICATION
# # #########################################################

# # labels = [
# #     "beverage bottle",
# #     "soft drink bottle",
# #     "beer bottle",
# #     "beverage can",
# #     "plastic bag",
# #     "plastic jar",
# #     "steel utensil",
# #     "steel glass",
# #     "cup",
# #     "food",
# #     "cleaning bottle"
# # ]

# # allowed = [
# #     "beverage bottle",
# #     "soft drink bottle"
# # ]

# # impure = False

# # for i, box in enumerate(unknown):

# #     x1, y1, x2, y2 = box

# #     x1 = max(0, x1)
# #     y1 = max(0, y1)
# #     x2 = min(W, x2)
# #     y2 = min(H, y2)

# #     crop = image[y1:y2, x1:x2]

# #     if crop.size == 0:
# #         continue

# #     crop = Image.fromarray(
# #         cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
# #     )

# #     inputs = processor(
# #         text=labels,
# #         images=crop,
# #         return_tensors="pt",
# #         padding=True
# #     ).to(DEVICE)

# #     with torch.no_grad():
# #         outputs = siglip(**inputs)
# #         probs = outputs.logits_per_image.softmax(dim=-1)
# #         idx = probs.argmax().item()

# #     label = labels[idx]

# #     print(f"Unknown {i+1}: {label}")

# #     if label not in allowed:
# #         impure = True

# # #########################################################
# # # FINAL RESULT
# # #########################################################

# # if impure:
# #     print("\nFINAL RESULT : IMPURE")
# # else:
# #     print("\nFINAL RESULT : PURE")



import os
import cv2
from ultralytics import YOLO

#############################################
# CONFIG
#############################################

INPUT_FOLDER = "downloaded_images"              # Folder containing input images
OUTPUT_FOLDER = "annotated_images"   # Folder to save annotated images

MODEL = "yolov8x.pt"                 # COCO pretrained model
CONFIDENCE = 0.25

#############################################
# CREATE OUTPUT FOLDER
#############################################

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#############################################
# LOAD MODEL
#############################################

print("Loading YOLO model...")
model = YOLO(MODEL)

# COCO class names
names = model.names

# Find refrigerator class id
refrigerator_class = None

for cls_id, cls_name in names.items():
    if cls_name.lower() == "refrigerator":
        refrigerator_class = cls_id
        break

if refrigerator_class is None:
    raise Exception("Refrigerator class not found!")

print(f"Refrigerator Class ID : {refrigerator_class}")

#############################################
# IMAGE EXTENSIONS
#############################################

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

images = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(extensions)
]

print(f"Found {len(images)} images\n")

#############################################
# PROCESS IMAGES
#############################################

for img_name in images:

    image_path = os.path.join(INPUT_FOLDER, img_name)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read {img_name}")
        continue

    results = model(
        image,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    detections = 0

    for box in results.boxes:

        cls = int(box.cls[0])

        # Ignore every class except refrigerator
        if cls != refrigerator_class:
            continue

        detections += 1

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        label = f"Refrigerator {confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )

        cv2.rectangle(
            image,
            (x1, y1 - th - 10),
            (x1 + tw + 8, y1),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            image,
            label,
            (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    output_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(output_path, image)

    print(f"{img_name:40s} -> {detections} refrigerator(s)")

print("\nDone!")
print(f"Annotated images saved to: {OUTPUT_FOLDER}")