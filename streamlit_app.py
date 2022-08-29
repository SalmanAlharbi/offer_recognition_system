import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import sys
sys.path.append("models\\research")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import pytesseract as tess
import re
import spacy
from spacy.matcher import Matcher
import base64
from io import BytesIO
from PIL import Image as im
page_detection_model = None
offer_category_index = None
offer_detection_model = None
data_category_index = None


def load_page_model():
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(
        r"my_models\models\offer_box_detection_model\pipeline.config")
    global page_detection_model
    page_detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=page_detection_model)
    ckpt.restore(
        r"my_models\models\offer_box_detection_model\ckpt-51").expect_partial()


def load_offer_model():
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(
        r"my_models\models\offer_data_recognition\pipeline.config")
    global offer_detection_model
    offer_detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=offer_detection_model)
    ckpt.restore(
        r"my_models\models\offer_data_recognition\ckpt-52").expect_partial()


def load_label_map_offer():
    data_category_index = label_map_util.create_category_index_from_labelmap(
        r"my_models\models\offer_data_recognition\label_map.pbtxt")
    return data_category_index


def load_label_map_page():
    page_category_index = label_map_util.create_category_index_from_labelmap(
        r"my_models\models\offer_box_detection_model\label_map.pbtxt")
    return page_category_index


@tf.function
def detect_fn_offer(image):
    global offer_detection_model
    image, shapes = offer_detection_model.preprocess(image)
    prediction_dict = offer_detection_model.predict(image, shapes)
    detections = offer_detection_model.postprocess(
        prediction_dict, shapes)
    return detections


@tf.function
def detect_fn_page(image):
    global page_detection_model
    image, shapes = page_detection_model.preprocess(image)
    prediction_dict = page_detection_model.predict(image, shapes)
    detections = page_detection_model.postprocess(
        prediction_dict, shapes)
    return detections


def local_css(css_path):
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def image_to_base64(image_array):
    buffered = BytesIO()
    image_array.save(buffered, format="png")
    image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64, {image_b64}"


def draw_detections(image_np, detections, label_map, min_score_thresh):
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+label_id_offset,
        detections['detection_scores'],
        label_map,
        use_normalized_coordinates=True,
        max_boxes_to_draw=-1,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        skip_labels=True,
    )
    return image_np_with_detections, detections['detection_boxes'], detections['detection_scores'], detections['detection_classes']


def detect_page(image_np):
    label_map = load_label_map_page()

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn_page(input_tensor)

    return draw_detections(image_np, detections, label_map, 0.8)


def detect_offer(image_np):
    label_map = load_label_map_offer()
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn_offer(input_tensor)
    return draw_detections(image_np, detections, label_map, 0.3)


def dim_rel_to_abs(image, box):
    box = tuple(box.tolist())
    ymin, xmin, ymax, xmax = box
    height, width, _ = image.shape
    ymin = int(ymin*height)
    xmin = int(xmin*width)
    ymax = int(ymax*height)
    xmax = int(xmax*width)
    return ymin, xmin, ymax, xmax


def expand_borders(ymin, ymax, xmin, xmax, image):
    height, width, _ = image.shape
    if(ymin-2 >= 0):
        ymin = ymin-2
    else:
        ymin = 0
    if(ymax+2 <= height):
        ymax = ymax+2
    else:
        ymax = height
    if(xmin-4 >= 0):
        xmin = xmin-4
    else:
        xmin = 0
    if(xmax+4 <= width):
        xmax = xmax+4
    else:
        xmax = width

    return ymin, ymax, xmin, xmax


def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((12, 6), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)


def remove_borders(image):
    contours, heiarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)


if __name__ == '__main__':
    config = r"--psm 6 --oem 3 "
    arabconfig = r"-l ara --psm 6 --oem 3 "
    qunconfig = r"--psm 6 --oem 3 "
    digitconfig = r"--psm 6 --oem 3 digits "
    local_css(r"assets\style.css")
    st.write("hello")
    file = st.file_uploader('Upload An Image', type=["jpg", "jpeg"])
    entered_items = st.empty()
    st.sidebar.write("Show Preproccing:")
    arabic_tit_cb = st.sidebar.checkbox("Arabic Title")
    english_tit_cb = st.sidebar.checkbox("English Title")
    Old_Cost_cb = st.sidebar.checkbox("Old Cost")
    New_Cost_cb = st.sidebar.checkbox("New Cost")
    quantity_cb = st.sidebar.checkbox("quantity")
    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )

    if(file):
        load_page_model()
        nlp = spacy.load("en_core_web_sm")
        img = im.open(file)
        image_np = image_np = np.array(img)
        image_np_with_detections,  boxes, scores, _ = detect_page(
            image_np)
        st.image(image_np_with_detections)
        st.markdown(
            "<hr />",
            unsafe_allow_html=True
        )
        offers = []
        load_offer_model()
        content = []
        for i in range(len(boxes)):
            if(scores[i] > 0.5):
                ymin, xmin, ymax, xmax = dim_rel_to_abs(
                    image_np_with_detections, boxes[i])

                ymin, ymax, xmin,  xmax = expand_borders(
                    ymin, ymax, xmin, xmax, image_np)
                offer_image_np = image_np[ymin:ymax, xmin:xmax]
                offer_image_np_with_detections,  offer_boxes, offer_scores, offers_labels = detect_offer(
                    offer_image_np)
                content = []

                height, width, _ = offer_image_np.shape
                for j in range(len(offer_boxes)):
                    if(offer_scores[j] > 0.3):
                        ymin, xmin, ymax, xmax = dim_rel_to_abs(
                            offer_image_np_with_detections, offer_boxes[j])
                        if(offers_labels[j] == 2):  # name in arabic
                            if arabic_tit_cb:
                                st.write("Arabic Title preprocessing")
                                st.image(offer_image_np[ymin:ymax, xmin:xmax])
                            ymin, ymax, xmin,  xmax = expand_borders(
                                ymin, ymax, xmin, xmax, offer_image_np)
                            data_image_np = offer_image_np[ymin:ymax,
                                                           xmin:xmax]
                            if arabic_tit_cb:
                                st.image(data_image_np)
                            data_image_np = cv2.resize(
                                data_image_np, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                            if arabic_tit_cb:
                                st.image(data_image_np)
                            content.append(
                                f"Arabic Title: "+tess.image_to_string(data_image_np, config=arabconfig))
                        elif(offers_labels[j] == 3):  # name in english
                            if english_tit_cb:
                                st.write("English Title preprocessing")
                                st.image(offer_image_np[ymin:ymax, xmin:xmax])
                            ymin, ymax, xmin,  xmax = expand_borders(
                                ymin, ymax, xmin, xmax, offer_image_np)
                            data_image_np = offer_image_np[ymin:ymax, xmin:xmax]
                            if english_tit_cb:
                                st.image(data_image_np)
                            data_image_np = cv2.resize(
                                data_image_np, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                            if english_tit_cb:
                                st.image(data_image_np)
                            english_title = tess.image_to_string(
                                data_image_np, config=config)
                            content.append("English Title: " + english_title)
                            doc = nlp(english_title.lower())
                            pattern = [{'DEP': 'ROOT'}]
                            matcher = Matcher(nlp.vocab)
                            matcher.add("item_name", [pattern])
                            mat_result = matcher(doc)

                            if mat_result:
                                # for simplsity and not having time lets just take the first root the mather found item_index[0]
                                rootIndex = mat_result[0][1]
                                item_name = doc[rootIndex].text
                                # clear tokens with dependency "dep" i.e. unclassified dependency
                                if(rootIndex > 0 and doc[rootIndex-1].dep_ in ["compound", "npadvmod", "nsubj"]):
                                    item_name = doc[rootIndex -
                                                    1].text+" "+item_name

                                if((rootIndex+1) < len(doc) and doc[rootIndex+1].dep_ in ["dobj", "compound"]):
                                    item_name = item_name + " " + \
                                        doc[rootIndex+1].text
                                content.append("Item Name: "+item_name)
                        elif offers_labels[j] == 4:  # qunitity

                            data_image_np = offer_image_np[ymin:ymax, xmin:xmax]
                            if quantity_cb:
                                st.image(data_image_np)
                            data_image_np = cv2.resize(
                                data_image_np, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            if quantity_cb:
                                st.image(data_image_np)
                            gray_image = cv2.cvtColor(
                                data_image_np, cv2.COLOR_BGR2GRAY)
                            if quantity_cb:
                                st.image(gray_image)
                            gray_image = remove_borders(gray_image)
                            if quantity_cb:
                                st.image(gray_image)
                            thresh, im_bw3 = cv2.threshold(
                                gray_image, 140, 255, cv2.THRESH_TOZERO)
                            if quantity_cb:
                                st.image(im_bw3)
                            temp = tess.image_to_string(
                                remove_borders(im_bw3), config=qunconfig)
                            content.append(
                                "Quintity: " + tess.image_to_string(remove_borders(im_bw3), config=qunconfig))

                        elif(offers_labels[j] == 0):  # old price
                            width = xmax-xmin
                            if Old_Cost_cb:
                                st.write("Old_Cost preprocessing")
                                st.image(offer_image_np[ymin:ymax, xmin:xmax])
                            data_image_np = offer_image_np[ymin:ymax,
                                                           int(xmin + width*0.1):int(xmax - width*0.1)]
                            if Old_Cost_cb:

                                st.image(data_image_np)
                            data_image_np = cv2.resize(
                                data_image_np, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
                            if Old_Cost_cb:
                                st.image(data_image_np)
                            gray_image = cv2.cvtColor(
                                data_image_np, cv2.COLOR_BGR2GRAY)
                            if Old_Cost_cb:
                                st.image(gray_image)
                            borders_removed = remove_borders(gray_image)
                            if Old_Cost_cb:
                                st.image(borders_removed)
                            thresh, im_bw2 = cv2.threshold(
                                borders_removed, 140, 255, cv2.THRESH_BINARY)
                            if Old_Cost_cb:
                                st.image(im_bw2)

                            padding = cv2.copyMakeBorder(
                                im_bw2, 0, 5, 0, 0, cv2.BORDER_CONSTANT, None, (255, 255, 255))

                            old_cost = tess.image_to_string(
                                padding, config=digitconfig)
                            if old_cost is not None:
                                old_cost = re.search("\d+[.]{1}\d+", old_cost)
                                if old_cost is not None:
                                    content.append("Old Price: " +
                                                   old_cost.group(0))

                        elif offers_labels[j] == 1:  # new price
                            if New_Cost_cb:
                                st.write("New_Cost preprocessing")
                                st.image(offer_image_np[ymin:ymax, xmin:xmax])
                            data_image_np = offer_image_np[ymin:ymax,
                                                           xmin+int((xmax-xmin)*0.3):xmax]
                            if New_Cost_cb:

                                st.image(data_image_np)
                            data_image_np = cv2.resize(
                                data_image_np, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            if New_Cost_cb:
                                st.image(data_image_np)
                            gray_image = cv2.cvtColor(
                                data_image_np, cv2.COLOR_BGR2GRAY)
                            if New_Cost_cb:
                                st.image(data_image_np)
                            bitwise = cv2.bitwise_not(gray_image)
                            if New_Cost_cb:
                                st.image(bitwise)
                            padding = cv2.copyMakeBorder(
                                bitwise, 5, 10, 5, 5, cv2.BORDER_CONSTANT, None, (255, 255, 255))
                            if New_Cost_cb:
                                st.image(padding)
                            thresh, im_bw = cv2.threshold(
                                padding, 100, 120, cv2.THRESH_BINARY)
                            if New_Cost_cb:
                                st.image(im_bw)
                            borders_removed = remove_borders(im_bw)
                            if New_Cost_cb:
                                st.image(borders_removed)
                            thin = thin_font(borders_removed)
                            if New_Cost_cb:
                                st.image(thin)
                            new_cost = tess.image_to_string(
                                thin, config=digitconfig)
                            if(new_cost is not None):
                                new_cost = re.search("\d+[.]{1}\d+", new_cost)
                                if(new_cost is not None):
                                    content.append(
                                        "New Price: " + new_cost.group(0))
                image_png = im.fromarray(
                    offer_image_np_with_detections)
                image_png = image_to_base64(image_png)

                st.markdown(
                    " ".join([
                        "<div class='r-text-recipe'>",
                        "<div class='food-title'>",
                        f"<h2 class='font-title text-bold'>Offer {j}</h2>",
                        "</div>",
                        '<div class="divider"><div class="divider-mask"></div></div>',
                        f"<img src='{image_png}' />",
                        "<h3 class='ingredients font-body text-bold'>Offer Data</h3>",
                        "<ul class='ingredients-list font-body'>",
                        " ".join(
                            [f'<li>{item}</li>' for item in content]),
                        "</ul>",
                        "</div>",
                        "<hr />"
                    ]),
                    unsafe_allow_html=True
                )
