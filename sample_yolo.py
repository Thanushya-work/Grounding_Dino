import os
import cv2
from ultralytics import YOLO

##########################################
# CONFIG
##########################################

INPUT_FOLDER = "shelf_images/shelf_images"              # Folder containing input images
OUTPUT_FOLDER = "annotated_images_yolo"   # Folder to save annotated images

CONF = 0.15

##########################################

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading YOLO-World...")

# Load YOLO-World
model = YOLO("yolov8x-worldv2.pt")

##########################################
# YOUR CLASSES
##########################################

classes = [

    "visicooler",

    "refrigerator",

    "beverage bottle",

    "soft drink bottle",

    "plastic bag",

    "plastic cover",

    "steel utensil",

    "steel glass",

    "plastic container",

    "food",

    "curd cup",

    "cleaning bottle",

    "juice carton",

    "beverage can"

]

model.set_classes(classes)

##########################################

extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

images = [

    f for f in os.listdir(INPUT_FOLDER)

    if f.lower().endswith(extensions)

]

print(f"Found {len(images)} images")

##########################################

for image_name in images:

    image_path = os.path.join(INPUT_FOLDER, image_name)

    image = cv2.imread(image_path)

    results = model.predict(

        source=image,

        conf=CONF,

        verbose=False

    )

    result = results[0]

    annotated = image.copy()

    print("\n", image_name)

    for box in result.boxes:

        cls = int(box.cls[0])

        label = classes[cls]

        conf = float(box.conf[0])

        print(label, round(conf, 2))

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = (0,255,0)

        cv2.rectangle(

            annotated,

            (x1,y1),

            (x2,y2),

            color,

            3

        )

        text = f"{label} {conf:.2f}"

        (tw,th),_ = cv2.getTextSize(

            text,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            2

        )

        cv2.rectangle(

            annotated,

            (x1,y1-th-10),

            (x1+tw+6,y1),

            color,

            -1

        )

        cv2.putText(

            annotated,

            text,

            (x1+3,y1-5),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0,0,0),

            2

        )

    cv2.imwrite(

        os.path.join(OUTPUT_FOLDER,image_name),

        annotated

    )

print("\nDone!")