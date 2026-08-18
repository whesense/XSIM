from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision as tv

from xsim.data.loaders import SensorData, load_camera_images_iterator
from xsim.utils import progress_bar


class SegFormerSegmentor:
    _semantic_classes = {
        'vehicle': [13, 14, 15],  # 'car', 'truck', 'bus'
        # 'person', 'rider', 'motorcycle', 'bicycle'
        'human': [11, 12, 17, 18],
        'sky': [10]
    }

    def __init__(
            self,
            name: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            device: str = 'cuda'
    ):
        from transformers import (
            SegformerImageProcessor,
            SegformerForSemanticSegmentation
        )
        self.name = name
        self.feature_extractor = SegformerImageProcessor.from_pretrained(name)
        self.model = (SegformerForSemanticSegmentation
                      .from_pretrained(name)
                      .to(device)
                      .eval())
        self.semantic_classes = {
            cls: torch.as_tensor(self._semantic_classes[cls]).long().to(device)
            for cls in self._semantic_classes
        }

    @torch.no_grad()
    def run_inference(self, image: torch.Tensor) -> torch.Tensor:
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].to(self.model.device)
        logits = self.model(**inputs).logits

        result = F.interpolate(logits, size=image.shape[:2], mode='bilinear')
        return result[0].argmax(dim=0)

    def human_mask(self, seg_map):
        return torch.isin(seg_map, self.semantic_classes['human'])

    def vehicle_mask(self, seg_map):
        return torch.isin(seg_map, self.semantic_classes['vehicle'])

    def sky_mask(self, seg_map):
        return torch.isin(seg_map, self.semantic_classes['sky'])

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        seg_map = self.run_inference(image)
        return torch.stack([
            self.human_mask(seg_map).byte().mul(255),
            self.vehicle_mask(seg_map).byte().mul(255),
            self.sky_mask(seg_map).byte().mul(255)
        ], dim=0)



def save_segmentation_masks(
        cameras: SensorData,
        out_path_fmt: str,
        segmentor_cls = SegFormerSegmentor,
        device: str = 'cuda'
):
    model = segmentor_cls(device=device)

    load_iter = load_camera_images_iterator(cameras, num_workers=8)

    with progress_bar('Segmenting images', total=cameras.total_num_images) as pbar:
        for (sensor_id, sample_idx), img in iter(load_iter):
            data = bytes(tv.io.encode_jpeg(model(img), quality=100).cpu().numpy())
            out_path = Path(out_path_fmt.format(
                sensor_id=sensor_id, sample_idx=sample_idx))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open('wb') as f:
                f.write(data)

            pbar.update()
