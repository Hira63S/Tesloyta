import argparse
import numpy as np
import time
import cv2
import torchvision
import torch
import torchvision.transforms as transforms
# arguments for the input, for the output
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help = 'path to input image')
ap.add_argument("-o", "--output", required=True, help = 'path to output image')
ap.add_argument("-m", "--model", required=False, help = 'path to the model')   # might not even need this because PyTorch
ap.add_argument("-l", "--labels", required=False, help= 'path to ImageNet labels (i.e., syn-sets)')
# directory would be:
# 'Desktop/Tesloyta/pi-deep-learning/models/synset_words.txt'
args = vars(ap.parse_args())

# read in the image
image = cv2.imread(args["image"])

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.483, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])

# labels

rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") +1:].split(",")[0] for r in rows]
for (i, idx) in enumerate(idxs):
    if i == 0:
        text = 'Label: {}, {:.2f}%'.format(classes[idx],
             preds[0][idx] * 100)
        cv2.putText(image, text, (5,25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

    print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
        classes[id], preds[0][idx]))

# blob = cv2.dnn.blobFromImage(image, 1, (227, 227), (104, 117, 123))

model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)
model.eval()

nn_input = transform(image)
input_batch = nn_input.unsqueeze(0)
output = model(input_batch)


# input_image = Image.open(filename)


cv2.imshow("Raw  Image", image)
cv2.waitKey(0)
