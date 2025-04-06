import os
import argparse
import cv2  # 需要安装 opencv-python: pip install opencv-python
import json
from pathlib import Path

from doclayout_yolo import YOLOv10
from tqdm import tqdm

# 假设这些导入和常量是您项目结构的一部分
# from magic_pdf.config.constants import MODEL_NAME
# from magic_pdf.model.model_list import AtomicModel
# from magic_pdf.model.sub_modules.model_init import AtomModelSingleton

# --- 您提供的模型初始化代码 ---
# 注意：这里我们直接使用您定义的类和初始化逻辑
# 在实际应用中，您可能需要调整模型路径和设备
# For demonstration, let's define placeholder constants if the original imports fail
try:
    from magic_pdf.config.constants import MODEL_NAME
    from magic_pdf.model.model_list import AtomicModel
    from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
except ImportError:
    print("Warning: magic_pdf modules not found. Using placeholder classes/variables.")


    class AtomicModel:
        Layout = "Layout"


    class MODEL_NAME:
        DocLayout_YOLO = "DocLayout_YOLO"


    class AtomModelSingleton:
        def __init__(self):
            self._models = {}

        def get_atom_model(self, atom_model_name, layout_model_name, doclayout_yolo_weights, device):
            # Simplified mock for demonstration if imports fail
            print(f"Mocking model loading for {layout_model_name} with weights {doclayout_yolo_weights} on {device}")
            # In a real scenario, this would load the actual model.
            # For this example, we will instantiate DocLayoutYOLOModel directly later.
            # We return a dummy object or None, and handle it in main.
            return None  # Indicate model loading needs to be handled differently

# --- 全局模型加载 (如果 magic_pdf 可用) ---
# 请确保权重路径 '/Users/wenruifeng/.cache/modelscope/hub/models/opendatalab/PDF-Extract-Kit-1___0/models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt'
# 和设备 'mps' 是正确的。如果 magic_pdf 不可用，我们将在 main 中直接实例化。
_WEIGHTS_PATH = '/Users/wenruifeng/.cache/modelscope/hub/models/opendatalab/PDF-Extract-Kit-1___0/models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt'
_DEVICE = 'mps'  # 或者 'cuda', 'cpu'

try:
    atom_model_manager = AtomModelSingleton()
    # 尝试使用您提供的 Singleton 加载模型
    # 注意：原始代码中 model = atom_model_manager.get_atom_model(...) 返回的是什么类型需要确认。
    # 假设它返回的是一个配置好的、可以调用的模型实例，或者 DocLayoutYOLOModel 实例。
    # 为了明确，我们将直接使用 DocLayoutYOLOModel 类。
    # global_model_instance = atom_model_manager.get_atom_model(
    #     atom_model_name=AtomicModel.Layout,
    #     layout_model_name=MODEL_NAME.DocLayout_YOLO,
    #     doclayout_yolo_weights=_WEIGHTS_PATH,
    #     device=_DEVICE,
    # )
    # If AtomModelSingleton doesn't return a ready-to-use model object of type DocLayoutYOLOModel,
    # initialize it directly.
    global_model_instance = None  # Placeholder
except NameError:  # Handle case where magic_pdf modules were not imported
    global_model_instance = None
    print("Skipping AtomModelSingleton initialization due to import errors.")
except Exception as e:
    print(f"Error during AtomModelSingleton initialization: {e}")
    global_model_instance = None


# --- 您提供的模型类定义 ---
class DocLayoutYOLOModel(object):
    def __init__(self, weight, device):
        print(f"Initializing YOLOv10 model with weights: {weight} on device: {device}")
        # Check if weight file exists
        if not os.path.exists(weight):
            raise FileNotFoundError(f"Model weight file not found at {weight}")
        self.model = YOLOv10(weight)
        self.device = device
        print("YOLOv10 model initialized successfully.")

    def predict(self, image):
        """
        Performs layout prediction on a single image.

        Args:
            image: A numpy array representing the image (loaded by cv2).

        Returns:
            A list of dictionaries, where each dictionary represents a detected
            layout element with 'category_id', 'poly', and 'score'.
        """
        layout_res = []
        if image is None:
            print("Error: Input image is None.")
            return layout_res

        print(f"Predicting layout for image with shape: {image.shape}")
        try:
            doclayout_yolo_res = self.model.predict(
                image,
                imgsz=1280,
                conf=0.10,
                iou=0.45,
                verbose=False,
                device=self.device
            )
            # Check if prediction returned results and access the first element
            if not doclayout_yolo_res:
                print("Warning: Model prediction returned empty list.")
                return layout_res

            prediction_output = doclayout_yolo_res[0]  # Access the Results object

            # Ensure boxes attribute exists and has data
            if prediction_output.boxes is None or len(prediction_output.boxes.xyxy) == 0:
                print("No bounding boxes detected.")
                return layout_res

            print(f"Detected {len(prediction_output.boxes.xyxy)} elements.")
            for xyxy, conf, cla in zip(
                    prediction_output.boxes.xyxy.cpu(),
                    prediction_output.boxes.conf.cpu(),
                    prediction_output.boxes.cls.cpu(),
            ):
                xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                new_item = {
                    "category_id": int(cla.item()),
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],  # Polygon format
                    "bbox": [xmin, ymin, xmax, ymax],  # Bbox format for drawing
                    "score": round(float(conf.item()), 3),
                }
                layout_res.append(new_item)
        except Exception as e:
            print(f"Error during model prediction: {e}")
            # import traceback
            # traceback.print_exc() # Print detailed traceback for debugging
        return layout_res

    def batch_predict(self, images: list, batch_size: int) -> list:
        # (Implementation kept as provided, but not used in the main function below)
        images_layout_res = []
        for index in tqdm(range(0, len(images), batch_size), desc="Layout Predict"):
            try:
                doclayout_yolo_res_batch = self.model.predict(
                    images[index: index + batch_size],
                    imgsz=1280,
                    conf=0.10,
                    iou=0.45,
                    verbose=False,
                    device=self.device,
                )
            except Exception as e:
                print(f"Error during batch prediction: {e}")
                continue  # Skip this batch on error

            for image_res in doclayout_yolo_res_batch:
                image_res_cpu = image_res.cpu()  # Move results to CPU once per image
                layout_res = []
                if image_res_cpu.boxes is None or len(image_res_cpu.boxes.xyxy) == 0:
                    images_layout_res.append(layout_res)  # Append empty list if no boxes
                    continue

                for xyxy, conf, cla in zip(
                        image_res_cpu.boxes.xyxy,
                        image_res_cpu.boxes.conf,
                        image_res_cpu.boxes.cls,
                ):
                    xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
                    new_item = {
                        "category_id": int(cla.item()),
                        "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                        "score": round(float(conf.item()), 3),
                    }
                    layout_res.append(new_item)
                images_layout_res.append(layout_res)

        return images_layout_res


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Perform document layout analysis on an image.")
    # parser.add_argument("image_path", type=str, default='/Users/wenruifeng/PycharmProjects/MinerU/page_0004.jpg',help="Absolute path to the input image file.")
    parser.add_argument("--weights", type=str, default=_WEIGHTS_PATH, help="Path to the YOLOv10 model weights file.")
    parser.add_argument("--device", type=str, default=_DEVICE,
                        help="Device to run inference on (e.g., 'cpu', 'cuda', 'mps').")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output files. Defaults to the input image's directory.")

    args = parser.parse_args()
    image_path = '/Users/wenruifeng/PycharmProjects/MinerU/page_0004.jpg'
    # Validate input image path
    image_path = Path(image_path)
    if not image_path.is_file():
        print(f"Error: Input image file not found at {image_path}")
        return

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # Define output file paths
    base_filename = image_path.stem
    output_json_path = output_dir / f"{base_filename}_layout.json"
    output_image_path = output_dir / f"{base_filename}_output.png"  # Save as PNG for potentially better quality

    print(f"Input image: {image_path}")
    print(f"Output JSON: {output_json_path}")
    print(f"Output image: {output_image_path}")
    print("-" * 30)

    # --- Model Initialization ---
    # Try using the globally loaded instance first, otherwise initialize directly.
    model_instance = global_model_instance
    if model_instance is None:
        print("Initializing DocLayoutYOLOModel directly...")
        try:
            model_instance = DocLayoutYOLOModel(weight=args.weights, device=args.device)
        except FileNotFoundError as e:
            print(e)
            return
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            return
    # elif not isinstance(model_instance, DocLayoutYOLOModel):
    #     # If the singleton returned something else, handle appropriately or raise error
    #     # For this example, we'll assume if it's not None, it's usable.
    #     # You might need to adapt this based on what AtomModelSingleton actually returns.
    #     print("Using model instance provided by AtomModelSingleton.")
    #     # Ensure it has a 'predict' method (duck typing)
    #     if not hasattr(model_instance, 'predict'):
    #          print("Error: Model instance from Singleton does not have a 'predict' method.")
    #          # Fallback to direct initialization if possible? Or just exit.
    #          print("Attempting direct initialization as fallback...")
    #          try:
    #              model_instance = DocLayoutYOLOModel(weight=args.weights, device=args.device)
    #          except Exception as e_fallback:
    #              print(f"Fallback initialization failed: {e_fallback}")
    #              return
    else:
        print("Using pre-initialized model instance.")
        # Ensure the existing instance has the correct device/weights if args differ?
        # For simplicity, we assume the pre-initialized one is desired.

    # --- Image Loading ---
    print("Loading image...")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Failed to load image from {image_path}. Check file format and permissions.")
        return
    print(f"Image loaded successfully (shape: {image.shape}).")

    # --- Prediction ---
    print("Running layout prediction...")
    layout_results = model_instance.predict(image)
    print(f"Prediction complete. Found {len(layout_results)} layout elements.")

    if not layout_results:
        print("No layout elements detected or prediction failed.")
        # Optionally save empty JSON and original image copy
        # with open(output_json_path, 'w') as f:
        #     json.dump([], f, indent=4)
        # cv2.imwrite(str(output_image_path), image)
        # print("Saved empty JSON and copy of original image.")
        return

    # --- Save JSON Output ---
    # Remove the 'bbox' key used only for drawing before saving JSON
    json_output_data = [{k: v for k, v in item.items() if k != 'bbox'} for item in layout_results]
    print(f"Saving layout results to {output_json_path}...")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output_data, f, indent=4, ensure_ascii=False)
        print("JSON saved successfully.")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

    # --- Draw Bounding Boxes and Save Output Image ---
    print(f"Drawing bounding boxes on image...")
    output_image = image.copy()  # Work on a copy

    # Optional: Define colors for different categories for better visualization
    # Example category names (replace with actual names for your model)
    category_names = {
        0: 'Text', 1: 'Title', 2: 'List', 3: 'Table', 4: 'Figure',
        # Add more categories as defined by your specific YOLO model's training
    }
    # Generate distinct colors (BGR format)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (192, 192, 192), (128, 0, 0), (0, 128, 0), (0, 0, 128)
    ]

    for item in layout_results:
        bbox = item.get('bbox')  # Use the bbox key we added
        if not bbox:
            print("Warning: Skipping item without bbox.")
            continue

        xmin, ymin, xmax, ymax = bbox
        category_id = item['category_id']
        score = item['score']

        color = colors[category_id % len(colors)]  # Cycle through colors
        label = category_names.get(category_id, f"ID:{category_id}")  # Get name or use ID
        display_text = f"{label}: {score:.2f}"

        # Draw rectangle
        cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)  # Thickness=2

        # Put label text
        # Calculate text size to put background
        (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Put filled rectangle as background for text
        cv2.rectangle(output_image, (xmin, ymin - text_height - baseline), (xmin + text_width, ymin), color,
                      -1)  # Filled
        # Put text itself
        cv2.putText(output_image, display_text, (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1)  # White text

    print(f"Saving annotated image to {output_image_path}...")
    try:
        success = cv2.imwrite(str(output_image_path), output_image)
        if success:
            print("Annotated image saved successfully.")
        else:
            print("Error: Failed to save annotated image.")
    except Exception as e:
        print(f"Error saving image file: {e}")

    print("-" * 30)
    print("Processing finished.")


# --- Entry Point ---
if __name__ == "__main__":
    main()