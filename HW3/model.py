# model.py
import torchvision

def get_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    return model

if __name__ == "__main__":
    model = get_model()
    print(model)
