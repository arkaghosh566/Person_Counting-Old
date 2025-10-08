import os
import cv2
import time
from multiprocessing import Process
from CrowdDet.tools.inference import run_inference

def process_rtsp_stream(model_dir, url):
    cv2.setUseOptimized(True)
    cv2.cuda.setDevice(0)
    run_inference(
        model_dir=model_dir,
        rstp_link=url
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    # ✅ Fixed camera URLs (added comma)
    camera_urls = [
        # "rtsp://172.14.0.187:554/rtsp/streaming?channel=01&subtype=0&transport=tcp", # OUT
        # "rtsp://admin:Capsi@24@172.14.0.111:554/?transport=tcp", 
        # "rtsp://admin:Capsi@24@172.14.0.112:554/?transport=tcp", # D,C
        # "rtsp://admin:Capsi@24@172.14.0.113:554/?transport=tcp", 
        # "rtsp://admin:Capsi@24@172.14.0.114:554/?transport=tcp", 
        # "rtsp://admin:Capsi@24@172.14.0.115:554/?transport=tcp", 
        # "rtsp://admin:Capsi@24@172.14.0.116:554/?transport=tcp", # B
        # "rtsp://admin:Capsi@24@172.14.0.117:554/?transport=tcp", # K,L
    #     "rtsp://admin:Capsi@24@172.14.0.118:554/?transport=tcp", 
    #     "rtsp://admin:Capsi@24@172.14.0.119:554/?transport=tcp", # C,D
    #     "rtsp://admin:Capsi@24@172.14.0.120:554/?transport=tcp", # B
    #     "rtsp://admin:Capsi@24@172.14.0.121:554/?transport=tcp", # L,K
    #     "rtsp://admin:Capsi@24@172.14.0.122:554/?transport=tcp", # M
    #     "rtsp://admin:Capsi@24@172.14.0.123:554/?transport=tcp", 
    #     "rtsp://admin:Capsi@24@172.14.0.124:554/?transport=tcp", # M
    #     "rtsp://admin:Capsi@24@172.14.0.125:554/?transport=tcp",
        # "rtsp://admin:Capsi@24@172.14.0.181:554/?transport=tcp",
    #     "rtsp://admin:Capsi@24@172.14.0.182:554/?transport=tcp",
    #     "rtsp://admin:Capsi@24@172.14.0.183:554/?transport=tcp",
        #   "rtsp://admin:Capsi@24@172.14.0.179:554/?transport=tcp",
        #   "rtsp://admin:Capsi@24@172.14.0.178:554/?transport=tcp",
          "rtsp://admin:Capsi@24@172.14.0.164:554/?transport=tcp",
          "rtsp://admin:Capsi@24@172.14.0.162:554/?transport=tcp",
        #   "rtsp://admin:Capsi@24@172.14.0.143:554/?transport=tcp"
    #     
    ]

    model_dir = r'D:\Arka\CrowdDet_Custom\CrowdDet\model\rcnn_emd_refine'

    # weights_list = [
    #                 r'D:\Arka\CrowdDet_Custom\CrowdDet\model\rcnn_emd_refine',
    #                 r'D:\Arka\CrowdDet_Custom\CrowdDet\model\rcnn_emd_simple',
    #                 r'D:\Arka\CrowdDet_Custom\CrowdDet\model\rcnn_fpn_baseline',
    #                 r'D:\Arka\CrowdDet_Custom\CrowdDet\model\retina_emd_simple',
    #                 r'D:\Arka\CrowdDet_Custom\CrowdDet\model\retina_fpn_baseline',
                # ]

    # os.makedirs('outputs', exist_ok=True)

    processes = []
    for url in camera_urls:
        p = Process(target=process_rtsp_stream, args=(model_dir, url))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # for weight_path in weights_list:
    #     print(f"\n🔁 Running inference with weight: {weight_path}")

    #     processes = []
    #     for url in camera_urls:
    #         p = Process(target=process_rtsp_stream, args=(weight_path, url))
    #         p.start()
    #         processes.append(p)

    #     print(f"⏳ Waiting 2 minutes for weight {weight_path}...")
    #     time.sleep(120)  # Let it run for 2 minutes

    #     for p in processes:
    #         if p.is_alive():
    #             p.terminate()
    #             p.join()

    #     print(f"✅ Finished with weight: {weight_path}")
