from loadmodel import LoadModel
from cropim import CropIm
from classifyim import Classify

print("I'm here")
model, lb = LoadModel()
def get_image():
    global model, lb
    img = CropIm("..\\testim\\acer_platanoides__03.jpg")
    label, prob = Classify(img, model, lb)
    print("success")
    result = label.replace("_", " ")
    result = result.capitalize()
    print("I am %d%% confident this is %s" % (prob, result))
get_image()

