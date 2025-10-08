import os
import json
import numpy as np
import cv2

color = {'green':(0,255,0),
        'blue':(255,165,0),
        'dark red':(0,0,139),
        'red':(0, 0, 255),
        'dark slate blue':(139,61,72),
        'aqua':(255,255,0),
        'brown':(42,42,165),
        'deep pink':(147,20,255),
        'fuchisia':(255,0,255),
        'yello':(0,238,238),
        'orange':(0,165,255),
        'saddle brown':(19,69,139),
        'black':(0,0,0),
        'white':(255,255,255)}

# def draw_boxes(img, boxes, scores=None, tags=None, line_thick=1, line_color='white'):
#     width = img.shape[1]
#     height = img.shape[0]
#     for i in range(len(boxes)):
#         one_box = boxes[i]
#         one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
#                     min(one_box[2], width - 1), min(one_box[3], height - 1)])
#         x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
#         cv2.rectangle(img, (x1,y1), (x2,y2), color[line_color], line_thick)
#         if scores is not None:
#             text = "{} {:.3f}".format(tags[i], scores[i])
#             cv2.putText(img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 0.5, color[line_color], line_thick)
#     return img

# def draw_boxes(img, boxes, scores=None, tags=None, line_thick=1, line_color='white'):
#     width = img.shape[1]
#     height = img.shape[0]
    
#     for i in range(len(boxes)):
#         one_box = boxes[i]
#         one_box = np.array([
#             max(one_box[0], 0), 
#             max(one_box[1], 0),
#             min(one_box[2], width - 1), 
#             min(one_box[3], height - 1)
#         ])
#         x1, y1, x2, y2 = np.array(one_box[:4]).astype(int)
        
#         # Draw bounding box
#         cv2.rectangle(img, (x1, y1), (x2, y2), color[line_color], line_thick)

#         # Draw triangle marker above the bounding box (pointing down)
#         triangle_center_x = (x1 + x2) // 2
#         triangle_top_y = max(y1 - 15, 0)
#         triangle_pts = np.array([
#             [triangle_center_x - 7, triangle_top_y],
#             [triangle_center_x + 7, triangle_top_y],
#             [triangle_center_x, triangle_top_y + 10]
#         ])
#         cv2.drawContours(img, [triangle_pts], 0, color[line_color], -1)  # filled triangle

#         # Draw label with score (optional)
#         if scores is not None and tags is not None:
#             text = "{} {:.3f}".format(tags[i], scores[i])
#             cv2.putText(img, text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[line_color], line_thick)

#     return img

# def draw_boxes(img, boxes, scores=None, tags=None, line_thick=1, line_color='white'):
#     width = img.shape[1]
#     height = img.shape[0]

#     for i in range(len(boxes)):
#         one_box = boxes[i]
#         one_box = np.array([
#             max(one_box[0], 0),
#             max(one_box[1], 0),
#             min(one_box[2], width - 1),
#             min(one_box[3], height - 1)
#         ])
#         x1, y1, x2, y2 = np.array(one_box[:4]).astype(int)

#         # Compute top-center of bounding box
#         triangle_center_x = (x1 + x2) // 2
#         triangle_top_y = max(y1 - 15, 0)

#         # Define triangle points
#         triangle_pts = np.array([
#             [triangle_center_x - 7, triangle_top_y],        # left
#             [triangle_center_x + 7, triangle_top_y],        # right
#             [triangle_center_x, triangle_top_y + 10]        # bottom center (pointing down)
#         ])

#         # Draw filled triangle
#         cv2.drawContours(img, [triangle_pts], 0, color[line_color], -1)

#         # Optional: Add score and tag above triangle
#         if scores is not None and tags is not None:
#             text = "{} {:.3f}".format(tags[i], scores[i])
#             cv2.putText(img, text, (x1, triangle_top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[line_color], line_thick)

#     return img

def draw_boxes(img, boxes, scores=None, tags=None, line_thick=1, line_color='white'):
    width = img.shape[1]
    height = img.shape[0]

    person_count = len(boxes)

    # Draw person count at top-left
    count_text = f"Person Count: {person_count}"
    cv2.putText(
        img,
        count_text,
        (10, 25),  # Position: top-left
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # Font scale (medium)
        (200, 200, 200),  # Light gray color
        2,  # Thickness
        cv2.LINE_AA
    )

    for i in range(person_count):
        one_box = boxes[i]
        one_box = np.array([
            max(one_box[0], 0),
            max(one_box[1], 0),
            min(one_box[2], width - 1),
            min(one_box[3], height - 1)
        ])
        x1, y1, x2, y2 = np.array(one_box[:4]).astype(int)

        # Compute top-center of bounding box
        triangle_center_x = (x1 + x2) // 2
        triangle_top_y = max(y1 - 15, 0)

        # Define triangle points
        triangle_pts = np.array([
            [triangle_center_x - 7, triangle_top_y],        # left
            [triangle_center_x + 7, triangle_top_y],        # right
            [triangle_center_x, triangle_top_y + 10]        # bottom center (pointing down)
        ])

        # Draw filled triangle
        cv2.drawContours(img, [triangle_pts], 0, color[line_color], -1)

        # Optional: Add score and tag above triangle
        if scores is not None and tags is not None:
            text = "{} {:.3f}".format(tags[i], scores[i])
            cv2.putText(
                img,
                text,
                (x1, triangle_top_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color[line_color],
                line_thick
            )

    return img


