import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# 本地运行时，请改成你电脑上的模型路径
# 例如：r"D:\models\vehicle_best.pt"
vehicle_model = YOLO("best_vehicle.pt")
lane_model = YOLO("best_lane.pt")

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def detect(image):
    if image is None:
        return None, "请上传图片"

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    vehicle_results = vehicle_model(img, conf=0.40)[0]
    lane_results = lane_model(img, conf=0.30)[0]

    annotated = img.copy()
    vehicle_count = 0
    other_count = 0

    if vehicle_results.boxes is not None:
        for box in vehicle_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = vehicle_results.names[cls_id]

            if cls_id in VEHICLE_CLASSES:
                vehicle_count += 1
                color = (0, 255, 0)
            else:
                other_count += 1
                color = (0, 165, 255)

            label = f"{name} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    lane_count = 0
    if lane_results.masks is not None:
        lane_count = len(lane_results.masks)
        for mask in lane_results.masks.data.cpu().numpy():
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            colored = np.zeros_like(img)
            colored[:, :, 2] = mask
            annotated = cv2.addWeighted(annotated, 1.0, colored, 0.45, 0)

    result_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    info = f"检测到车辆：{vehicle_count} 辆 | 其他目标：{other_count} 个 | 车道线：{lane_count} 条"
    return result_img, info

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil", label="上传路况图片"),
    outputs=[
        gr.Image(type="numpy", label="检测结果"),
        gr.Textbox(label="检测信息")
    ],
    title="🚗 车辆检测 + 车道线分割系统",
    description="绿色框 = 车辆 | 橙色框 = 其他目标 | 红色区域 = 车道线"
)

if __name__ == "__main__":
    demo.launch()
