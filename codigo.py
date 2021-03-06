#Importamos todas las librerías necesarias
import torch
import torch.optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn as nn

import os
import numpy
import PIL

from tqdm.notebook import tqdm

#Importamos la CNN de los modelos de torchvision
from torchvision.models import mobilenet_v3_small

mobilenet_v3_small = mobilenet_v3_small(pretrained=True)

#Para emplear archivos de Drive como su fuesen una carpeta local, desde el Google Collab
from google.colab import drive
drive.mount('/content/drive')

batch_size = 32

# Cargamos las imagenes por carpeta
train_dataset = ImageFolder('/content/drive/MyDrive/Tesis/Galería /Train',
                      # transforms.Compose([transforms.Resize((64, 64)),
                      # transforms.Grayscale(1),
                      transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor()]))
test_dataset = ImageFolder('/content/drive/MyDrive/Tesis/Galería /Test',
                      # transforms.Compose([transforms.Resize((64, 64)),
                      # transforms.Grayscale(1),
                      transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor()]))

# Organizamos los datos en lotes y los mezclamos automáticamente
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Cargamos el modelo y empleamos los recursos de GPU para accelerar el aprendizaje
model= mobilenet_v3_small.cuda()
criterio = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1)


for epoch in range(10):
  train_loss = 0
  for X, y in tqdm(train_dataloader):
    X = X.cuda()
    y = y.cuda()

    y_hat = model(X)
    # Las salidas son un paso antes del criterio de entropía cruzada

    loss = criterio(y_hat, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # Actualiza los parametros

    # Acumulamos los errores por cada batch
    # Usamos la función detach para que no se almacenen los gradientes de esta suma
    train_loss += float(loss.detach())

  print('L:', train_loss/len(train_dataloader))
  
#Para poder visualizar las imágenes  
import matplotlib.pyplot as plt
import numpy as np

batch_index = 0
#Cargamos los datos de imágenes y labels
xs, ys = list(train_dataloader)[batch_index]

img_index = 0 #Elegir un número del 0 al 31
x_img = xs[img_index, 0].numpy()

#Mostramos una imagen de un atractor y su respectiva etiqueta
plt.imshow(x_img, cmap='gray')
plt.show()
print(f'Tipo de atractor: {int(ys[img_index])} (1: regular, 0:caótico)')

#Calculamos e imprimimos la precisión de la CNN
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    

    for images, labels in train_dataloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Nb of samples = {n_samples}, nb of correctly predicted = {n_correct}')
    print(f'Accuracy of the network on train examples: {acc} %')
    n_correct = 0
    n_samples = 0
    
    for images, labels in test_dataloader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Nb of samples = {n_samples}, nb of correctly predicted = {n_correct}')
    print(f'Accuracy of the network on test examples: {acc} %')
   
   
#Se realiza la prueba con imágenes test. 
batch_index = 0
xs, ys = list(test_dataloader)[batch_index] 
xs = xs.cuda()
ys = ys.cuda()

image_index = 16

outputs = model(xs)
_, predicted = torch.max(outputs, 1)
print(predicted)
print(ys) 

xs = xs.cpu()
x_img = xs[image_index, 0].numpy()
plt.imshow(x_img, cmap='gray')
plt.show()
print(f'Atractor número {image_index + 1}')
print(f'Tipo de atractor: {int(ys[image_index])} (1: regular, 0:caótico)')
xs = xs.cuda()

