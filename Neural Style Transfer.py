# importing the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image


# creating a class VGG to get features from the intermediate layers

class VGG(nn.Module):

  def __init__(self):
    super(VGG,self).__init__()

    # choosen layer to get output we have to get the features from the conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    self.chosen_layers=['0','5','10','19','28']

    # getting till 28th layer of the pretrained vgg model
    self.model=models.vgg19(pretrained=True).features[:29]


  def forward(self,x):
    features=[]
    # passing our inputs to the vgg layer by layer
    for layer_num,layer in enumerate(self.model):
      x=layer(x)

      if str(layer_num) in self.chosen_layers:
        features.append(x)

    return features


# utility function to load image
def load_img(img_name):
    image=Image.open(img_name)
    image=loader(image).unsqueeze(0)
    return image.to(device)

# setting the device to cpu
device=torch.device("cpu")
img_size=356
loader=transforms.Compose([ transforms.Resize((img_size,img_size)),
                           transforms.ToTensor()])

# defining the path for the content_image and style image
content_image_path="drive/MyDrive/annahathaway.png"
style_image_path="drive/MyDrive/style.jpg"

def stylize(content_image_path,style_image_path,total_steps=6000,learning_rate=0.001,alpha=1,beta=0.01):

    # loading the content and style image
    content_image=load_img(content_image_path)
    style_image=load_img(style_image_path)

    # creating the model in the evaluation mode
    model=VGG().to(device).eval()

    # creating generated image as a  copy of content image
    generated_image=content_image.clone().requires_grad_(True)

    # setting hyper parameters
    TOTAL_STEPS=total_steps
    LEARNING_RATE=learning_rate
    ALPHA=alpha
    BETA=beta

    # defining the Adam optimizer to optimize the generated_image
    optimizer=optim.Adam([generated_image],lr=LEARNING_RATE)

    # traning loop
    for step in range(TOTAL_STEPS):

        # passing the content_image, style_image, generated_image throught the VGG
        content_features=model(content_image)
        style_features=model(style_image)
        generated_features=model(generated_image)

        # defining the style loss
        style_loss=0
        content_loss=0

        for gen_feature,content_feature,style_feature in zip(generated_features,content_features,style_features):
            # getting the shapes
            batch_size,channel,height,width=gen_feature.shape

            # calculating the content loss
            content_loss+=torch.mean((gen_feature-content_feature)**2)

            """ here we have 1 image so batch is 1 therefore g=(channel,height,width) ===> g'=(channel, height*width)
                Gram matrix G=g'.g'T
                g'= (channel,height*width) , g'T= (height*width,channel)
                G=  (channel,channel)
            """
            # calculating the gram matrix for generated image
            G=gen_feature.view(channel,height*width).mm( gen_feature.view(channel,height*width).t() )
            # calculating the gram matrix for the style image
            S=style_feature.view(channel,height*width).mm( style_feature.view(channel,height*width).t() )
            # computing the style loss
            style_loss+=torch.mean((G-S)**2)

        # computing the total loss
        total_loss= (ALPHA*content_loss) + (BETA*style_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    if step%10==0:
        print(f"The total loss at step{step} is {total_loss}")
        save_image(generated_image,f"generated_image_{step}.png")

  


