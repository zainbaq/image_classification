import torch
import torch.nn.functional as F
from torchvision import transforms as T
import cv2 as cv
import argparse
from modules.utils import load_checkpoint
from modules.model import ImageClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, nargs="?", default='checkpoints/checkpoint.pt')
parser.add_argument('--image', type=str, required=True)
args = parser.parse_args()

def main():
    transform = T.Compose([
        T.ToTensor(),
        T.Grayscale(),
        T.Normalize([0], [1])
    ])
    image = transform(cv.imread(args.image))

    model = ImageClassifier(10, 64)
    model = load_checkpoint(args.model, model)

    output = model(image.unsqueeze(0))
    prediction = torch.argmax(F.softmax(output, dim=1)).item()
    print(prediction)

if __name__ == '__main__':
    main()
