from django.shortcuts import render
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import json


# Create your views here.
img_height, img_width=224,224
with open('./models/imagenet_class_index.json','r') as f:
    labelInfo=f.read()

labelInfo=json.loads(labelInfo)

model = models.densenet121(pretrained=True)
model.eval()


def transform_image(testimage):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    image = Image.open(testimage)
    return my_transforms(image).unsqueeze(0)


def get_prediction(testimage):
    tensor = transform_image(testimage=testimage)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return labelInfo[predicted_idx]



def index(request):
    context={'a':1}
    return render(request,'index.html',context)



def predictImage(request):

    fileObj=request.FILES['filePath']  # sumatran-tiger-wz-gsmp-m.jpg
    fs=FileSystemStorage()   #save img into system
    filePathName=fs.save(fileObj.name,fileObj) #save the file in root directory,but add media, it will save in media folder
    filePathName=fs.url(filePathName)  #give the full file path, media/sumatran-tiger-wa-gsmp-m_NT!LDhb.jpg
    testimage='.'+filePathName


    _,predictedLabel=get_prediction(testimage=testimage)


    context={'filePathName':filePathName,'predictedLabel':predictedLabel}
    return render(request,'index.html',context)

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context)