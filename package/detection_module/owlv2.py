import requests
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class Owlv2:
    def __init__(self, model_id="google/owlv2-base-patch16-ensemble", device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)

    def load_image_from_url(self, image_url):
        return Image.open(requests.get(image_url, stream=True).raw)

    def detect_objects(
        self, 
        image_path,
        object_names,
        box_threshold = 0.5,
    ):
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text           = object_names, 
            images         = image, 
            return_tensors = "pt",
            ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs      = outputs,
            target_sizes = torch.Tensor([image.size[::-1]]),
            threshold    = box_threshold,
        )
        # NOTE : https://github.com/huggingface/transformers/blob/573565e35a5cc68f6cfb6337f5a93753ab16c65b/src/transformers/models/owlv2/image_processing_owlv2.py#L484
        # I guess post_process_object_detection has no nms (non-maximum suppression) and it returns all the boxes.
        result = results[0]
        result['object_names'] = object_names
        result['n'] = len(object_names)
        
        return results[0]

def plot_detection_result(
        image            = None,
        image_path       = '', 
        detection_result = {}, 
        figsize          = (8,6),
        fontsize         = 10,
    ):
    """ 
        Plot detection result
    """
    # Parse
    if image is None: image = plt.imread(image_path) 
    object_names = detection_result['object_names']
    
    # Plot
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image) # show image
    for i, box in enumerate(detection_result['boxes']):
        score = detection_result['scores'][i]
        label = detection_result['labels'][i]
        (x1, y1, x2, y2) = box
        # 바운딩 박스 그리기
        rect = patches.Rectangle(
            (x1, y1), 
            x2-x1, 
            y2-y1,
            linestyle = '--',
            linewidth = 1,
            edgecolor = 'r', 
            facecolor = 'none',
        )
        ax.add_patch(rect)
        object_name = object_names[label]
        # 객체 이름과 점수 표시
        ax.text(
            x1, y1, 
            f'{object_name}: {score:.2f}', 
            bbox     = dict(lw=0.1,fc='yellow',alpha=0.5,pad=0), 
            fontsize = fontsize,
            color    = 'black',
            va       = 'bottom', 
            ha       = 'left',
        )
    plt.axis('off')
    plt.show()
