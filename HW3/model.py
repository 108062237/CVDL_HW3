
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights
)
def build_model(cfg):
   
    backbone = cfg['MODEL'].get('BACKBONE', 'resnet50')
    pretrained = cfg['MODEL'].get('PRETRAINED', True)
    num_classes = cfg['MODEL']['NUM_CLASSES']

    # Select Mask R-CNN with specified backbone
    if backbone == 'resnet50':
        model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,               
            box_score_thresh=0.05,          
        )
    elif backbone in ['resnet101', 'resnet152',
                           'resnext50_32x4d', 'resnext101_32x8d',
                           'wide_resnet50_2', 'wide_resnet101_2']:
        backbone = resnet_fpn_backbone(backbone, pretrained=True)

        model = MaskRCNN(backbone, num_classes=5) 
    else:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose 'resnet50' or 'resnet101'.")

    # Replace the box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor head
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, hidden_layer, num_classes)

    return model


if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open('configs/config.yaml', 'r'))
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_params / 1e6:.2f}M")
    # Print summary of model parts
    # print(f"Built Mask R-CNN with backbone: {cfg['MODEL'].get('BACKBONE', 'resnet50')}")
    # print(model.roi_heads)
