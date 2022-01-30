#########
from email.mime import image
from flask import Flask, redirect,render_template,request,redirect,jsonify,make_response
import os                       # for working with files
import numpy as np              # for numerical computationss
import torch                    # Pytorch module 
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
from PIL import Image           # for checking images
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import glob
from datetime import datetime
import json

#########
# for calculating the accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# base class for the model
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)
# resnet architecture 
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out 

def predict_image(img, model):
    """Converts image to array and return the predicted class
        with highest probability"""
    # Convert to a batch of 1
    xb = img.unsqueeze(0)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label

    return preds[0].item()
#########

pd_classes = ['Apple___Apple_scab',
 'Apple___Black_rot',
  'Apple___Cedar_apple_rust',
  'Apple___healthy',
   'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
     'Cherry_(including_sour)___healthy',
      'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
       'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
         'Grape___Black_rot',
          'Grape___Esca_(Black_Measles)',
           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
             'Orange___Haunglongbing_(Citrus_greening)',
              'Peach___Bacterial_spot',
               'Peach___healthy',
               'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy',
                 'Potato___Early_blight',
                 'Potato___Late_blight',
                  'Potato___healthy',
                   'Raspberry___healthy',
                    'Soybean___healthy',
                     'Squash___Powdery_mildew',
                      'Strawberry___Leaf_scorch',
                       'Strawberry___healthy',
                        'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight',
                         'Tomato___Late_blight',
                          'Tomato___Leaf_Mold',
                           'Tomato___Septoria_leaf_spot',
                            'Tomato___Spider_mites Two-spotted_spider_mite',
                             'Tomato___Target_Spot',
                              'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                               'Tomato___Tomato_mosaic_virus',
                                'Tomato___healthy']
################
model = torch.load('pd_model.pt',map_location=torch.device('cpu'))
temp_dir = 'static\\images\\uploads'
###############
prediction = ''
test_images = []
label = ''
disname = ''
dissol1 = ''
dissol2 = ''
dissol3 = ''
###############
f = open('data.json')
disdata= json.load(f)

################
def process():
    t = glob.glob(temp_dir + '\\imgs\\'+"*")[0]
    ###############
    image = Image.open(t)
    image.thumbnail((256,256), Image.ANTIALIAS)
    image.save(temp_dir+'\\temp\\temp_pic.jpg')
    ###############
    test = ImageFolder(temp_dir, transform=transforms.ToTensor())
    ###############
    test_images = sorted(os.listdir(temp_dir + '\\temp')) # since images in test folder are in alphabetical order
    img, label = test[0]
    prediction = str(predict_image(img, model))
    ##############
    disname = disdata[prediction][0]['name']
    dissol1 = disdata[prediction][0]['solution1']
    dissol2 = disdata[prediction][0]['solution2']
    dissol3 = disdata[prediction][0]['solution3']
    ##############
    os.remove(t)
    os.remove(temp_dir+'\\temp\\temp_pic.jpg')
    ##############
    return disname, dissol1, dissol2, dissol3




app = Flask(__name__)


@app.route("/",methods=["POST","GET"])
def home():
    if request.method=="POST":
        return render_template("product.html")
    
    return render_template("index.html")

@app.route("/about")
def about():
    
    return render_template("aboutus.html")

@app.route("/diseases")
def diseases():
    
    return render_template("disease.html")
    
@app.route("/description")
def description():
    
    return render_template("description.html")

@app.route("/results")
def soln():
    return render_template("result.html")


app.config["IMAGE_UPLOADS"]=("static\\images\\uploads\\imgs")

@app.route("/products",methods=["POST","GET"])
def products():
    if request.method=="POST":
        if request.files:
            image=request.files["image"]
            image.save(os.path.join(app.config["IMAGE_UPLOADS"],image.filename))
            disname, dissol1, dissol2, dissol3 = process()
            return render_template("result.html",disease=disname,solution1=dissol1,solution2=dissol2,solution3=dissol3)
        
    return render_template("product.html")
    


if __name__ == "__main__":
    app.run(debug=True)
