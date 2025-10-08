import os
import sys
import argparse

import cv2
import torch
import numpy as np

sys.path.insert(0, '../lib')
from CrowdDet.lib.utils import misc_utils, visual_utils, nms_utils

def inference(args, config, network):
    cv2.setUseOptimized(True)
    
    misc_utils.ensure_dir('outputs')

    model_name = os.path.basename(args.model_dir) 
    model_file = os.path.join(args.model_dir, f"{model_name}.pth")
    assert os.path.exists(model_file)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    torch.backends.cudnn.benchmark = True

    # build and load model
    net = network().to(device)
    net.eval()
    check_point = torch.load(model_file, map_location=device)
    net.load_state_dict(check_point['state_dict'])

    # Open RTSP stream
    cap = cv2.VideoCapture(args.rstp_link)
    if not cap.isOpened():
        print(f"❌ Unable to open RTSP stream: {args.rstp_link}")
        return

    print("✅ RTSP stream opened. Running inference...")

    frame_count = 5

    cv2.namedWindow("CrowdDet Inference", cv2.WINDOW_NORMAL)

    # Resize the image just for display
    cv2.resizeWindow("CrowdDet Inference", (480, 270))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame.")
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue  # Skip this frame

        # Preprocess the frame
        try:
            image, resized_img, im_info = get_data(
                frame, config.eval_image_short_size, config.eval_image_max_size, device
            )
            resized_img = resized_img.to(device)
            im_info = im_info.to(device)
        except Exception as e:
            print(f"⚠️ Error during preprocessing: {e}")
            continue

        # Run model inference
        with torch.no_grad():
            pred_boxes = net(resized_img, im_info).cpu().numpy()

        pred_boxes = post_process(pred_boxes, config, im_info[0, 2].item())
        pred_tags = pred_boxes[:, 5].astype(np.int32).flatten()
        pred_tags_name = np.array(config.class_names)[pred_tags]

        # Draw results
        image = visual_utils.draw_boxes(
            image,
            pred_boxes[:, :4],
            scores=pred_boxes[:, 4],
            tags=pred_tags_name,
            line_thick=1,
            line_color='red'
        )

        # cv2.imshow("CrowdDet Inference", image)

        cv2.imshow("CrowdDet Inference",image)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
    
def post_process(pred_boxes, config, scale):
    if config.test_nms_method == 'set_nms':
        assert pred_boxes.shape[-1] > 6, "Not EMD Network! Using normal_nms instead."
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        top_k = pred_boxes.shape[-1] // 6
        n = pred_boxes.shape[0]
        pred_boxes = pred_boxes.reshape(-1, 6)
        idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
        pred_boxes = np.hstack((pred_boxes, idents))
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_utils.set_cpu_nms(pred_boxes, 0.5)
        pred_boxes = pred_boxes[keep]
    elif config.test_nms_method == 'normal_nms':
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
        keep = nms_utils.cpu_nms(pred_boxes, config.test_nms)
        pred_boxes = pred_boxes[keep]
    elif config.test_nms_method == 'none':
        assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
        pred_boxes = pred_boxes.reshape(-1, 6)
        keep = pred_boxes[:, 4] > config.pred_cls_threshold
        pred_boxes = pred_boxes[keep]
    #if pred_boxes.shape[0] > config.detection_per_image and \
    #    config.test_nms_method != 'none':
    #    order = np.argsort(-pred_boxes[:, 4])
    #    order = order[:config.detection_per_image]
    #    pred_boxes = pred_boxes[order]
    # recovery the scale
    pred_boxes[:, :4] /= scale
    keep = pred_boxes[:, 4] > config.visulize_threshold
    pred_boxes = pred_boxes[keep]
    return pred_boxes

def get_data(image, short_size, max_size, device):
    resized_img, scale = resize_img(image, short_size, max_size)

    original_height, original_width = image.shape[0:2]
    height, width = resized_img.shape[0:2]
    resized_img = resized_img.transpose(2, 0, 1)
    im_info = np.array([height, width, scale, original_height, original_width, 0], dtype=np.float32)

    resized_img_tensor = torch.from_numpy(np.expand_dims(resized_img, axis=0)).float()
    im_info_tensor = torch.from_numpy(np.expand_dims(im_info, axis=0)).float()

    return image, resized_img_tensor.to(device), im_info_tensor.to(device)


def resize_img(image, short_size, max_size):
    height = image.shape[0]
    width = image.shape[1]
    im_size_min = np.min([height, width])
    im_size_max = np.max([height, width])
    scale = (short_size + 0.0) / im_size_min
    if scale * im_size_max > max_size:
        scale = (max_size + 0.0) / im_size_max
    t_height, t_width = int(round(height * scale)), int(
        round(width * scale))
    resized_image = cv2.resize(
            image, (t_width, t_height), interpolation=cv2.INTER_LINEAR)
    return resized_image, scale

# def run_inference():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
#     parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
#     parser.add_argument('--img_path', '-i', default=None, required=True, type=str)
#     args = parser.parse_args()
#     # import libs
#     model_root_dir = os.path.join('../model/', args.model_dir)
#     sys.path.insert(0, model_root_dir)
#     from config import config
#     from network import Network
#     inference(args, config, Network)

def  run_inference(model_dir, rstp_link):
    # Import libs dynamically from model_dir
    model_root_dir = os.path.join('../model/', model_dir)
    sys.path.insert(0, model_root_dir)

    from config import config
    from network import Network

    # Create a dummy args-like object
    class Args:
        def __init__(self, model_dir, rstp_link):
            self.model_dir = model_dir
            self.rstp_link = rstp_link

    args = Args(model_dir, rstp_link)

    inference(args, config, Network)

if __name__ == '__main__':
    run_inference()
