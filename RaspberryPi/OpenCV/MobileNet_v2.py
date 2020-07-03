import cv2
import numpy as np
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help = "input image file")
ap.add_argument("-o", "--output", required=True,
    help="output file")
ap.add_argument("-l", "--labels", required=True,
    help="labels file")
args = vars(ap.parse_args())

# read in the labels file and extract classes names
rows = open('synset_words.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

#def transform_img(image):
image = Image.open(args["input"])
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Load the Model

mobile_net = torchvision.models.mobilenet_v2(pretrained=True)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    mobile_net.to('cuda')

with torch.no_grad():
    start = time.time()
    mobile_net.eval()
    preds = mobile_net(input_batch)
    end = time.time()
    print("[INFO] classification time {:.5} seconds".format(end - start))

# convert the preds into probabilities
probs = torch.nn.functional.softmax(preds[0], dim=0)

sort_probs, indices = torch.sort(probs, descending=True)
sorted_probs = sort_probs[:5]

for i, idx in enumerate(sorted_probs):
    if i == 0:
        text = "Label: {}, {:.2f}%".format(classes[int(torch.argmax(probs))],
                                           int(sorted_probs[0] * 100))
        print(classes[int(torch.argmax(probs))])
        cv2.putText(image, text, (100,250), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
        prob = float(sorted_probs[0] * 100)

        print("[INFO] {}. label: {}, probability: {:.5}".format(i+1, text, prob))


cv2.imshow("image", image)
cv2.waitKey(0)
