import torch 
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time 
import torch.utils.data

#%% device config EXTRA DEFAULT CPU but GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:",device)

def read_images(path, num_img):
    array = np.zeros((num_img,64*32))
    i = 0 
    for img in os.listdir(path):
        img_path = path + "\\" + img
        img = Image.open(img_path, mode = "r")
        data = np.asarray(img, dtype = "uint8")
        data = data.flatten()
        array[i, :] = data
        i += 1
    return array    

#read train negative
train_neg_path = r"D:/Work/Advanced deep learning 5.1/LSIFIR/Classification/Train/neg"
train_neg_num_img = 43390
train_neg_array = read_images(train_neg_path,train_neg_num_img)

x_train_neg_tensor = torch.from_numpy(train_neg_array).view(train_neg_num_img, 1, 64, 32).float()
print("x_train_neg_tensor:",x_train_neg_tensor.size())

y_train_neg_tensor = torch.zeros(train_neg_num_img, dtype = torch.long)
print("y_train_neg_tensor:",y_train_neg_tensor.size())

#read train positive

train_pos_path = r"D:/Work/Advanced deep learning 5.1/LSIFIR/Classification/Train/pos"
train_pos_num_img = 10208
train_pos_array = read_images(train_pos_path,train_pos_num_img)

x_train_pos_tensor = torch.from_numpy(train_pos_array).view(train_pos_num_img, 1, 64, 32).float()
print("x_train_pos_tensor:",x_train_pos_tensor.size())

y_train_pos_tensor = torch.ones(train_pos_num_img, dtype = torch.long)
print("y_train_pos_tensor:",y_train_pos_tensor.size())

#%% concat train
x_train = torch.cat((x_train_neg_tensor, x_train_pos_tensor),0)
y_train = torch.cat((y_train_neg_tensor, y_train_pos_tensor),0)
print("x_train:",x_train.size())
print("y_train:",y_train.size())

#------------------------------------------------------------------------------------------------------
#read test negative 22050
test_neg_path = r"D:/Work/Advanced deep learning 5.1/LSIFIR/Classification/Test/neg"
test_neg_num_img = 22050
test_neg_array = read_images(test_neg_path,test_neg_num_img)

x_test_neg_tensor = torch.from_numpy(test_neg_array).view(test_neg_num_img, 1, 64, 32).float()
print("x_test_neg_tensor:",x_test_neg_tensor.size())

y_test_neg_tensor = torch.zeros(test_neg_num_img, dtype = torch.long)
print("y_test_neg_tensor:",y_test_neg_tensor.size())

#read test positive 5944

test_pos_path = r"D:/Work/Advanced deep learning 5.1/LSIFIR/Classification/Test/pos"
test_pos_num_img = 5944
test_pos_array = read_images(test_pos_path,test_pos_num_img)

x_test_pos_tensor = torch.from_numpy(test_pos_array).view(test_pos_num_img, 1, 64, 32).float()
print("x_test_pos_tensor:",x_test_pos_tensor.size())

y_test_pos_tensor = torch.ones(test_pos_num_img, dtype = torch.long)
print("y_test_pos_tensor:",y_test_pos_tensor.size())

#%% concat train
x_test = torch.cat((x_test_neg_tensor, x_test_pos_tensor),0)
y_test = torch.cat((y_test_neg_tensor, y_test_pos_tensor),0)
print("x_test:",x_test.size())
print("y_test:",y_test.size())

#%% Visualize 
plt.imshow(x_train[45003,:].reshape(64,32), cmap="gray")

#%% CNN Model

#hyperparameters

num_epochs = 100
num_classes = 2 
batch_size = 64
learning_rate = 0.00001

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        
        self.fc1 = nn.Linear(16*5*13, 520)  #16*13*5
        self.fc2 = nn.Linear(520,130)
        self.fc3 = nn.Linear(130,num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 16*13*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
        
   
train = torch.utils.data.TensorDataset(x_train, y_train)
trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

test = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)   

net = Net()
if torch.cuda.is_available():
    net = net.to(device)
# net = Net().to(device) send to GPU if using a GPU this module 

#%% Loss and Optimizer

criterion = nn.CrossEntropyLoss()

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = 0.8)

#%% Train a network
start = time.time()

train_acc = []
test_acc = []
loss_list = []

use_gpu = True if torch.cuda.is_available() else False  #if use a graphic card you must change to 'True'

for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device) if use_gpu else inputs
        labels = labels.to(device) if use_gpu else labels

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        
    #test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device) if use_gpu else images
            labels = labels.to(device) if use_gpu else labels
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
    acc1 = 100*correct/total       
    print("accuracy test:",acc1)       
    test_acc.append(acc1)
     
    #train
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.to(device) if use_gpu else images
            labels = labels.to(device) if use_gpu else labels
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
           
            
            
    acc2 = 100 * correct/total       
    print("accuracy train:",acc2)       
    train_acc.append(acc2)
      
print("Train is Done !")      
     
    
end = time.time()
process_time = (end - start)/60
print("process time:",process_time)



#%% Saving Training Model

torch.save(net.state_dict(), 'trained_model.pth')
print("Trained model saved.")

#%% visualize
fig, ax1 = plt.subplots()

plt.plot(loss_list,label = "Loss",color = "black")

ax2 = ax1.twinx()

ax2.plot(np.array(test_acc)/100,label = "Test Acc",color="green")
ax2.plot(np.array(train_acc)/100,label = "Train Acc",color= "red")
ax1.legend()
ax2.legend()
ax1.set_xlabel('Epoch')
fig.tight_layout()
plt.title("Loss vs Test Accuracy")
plt.show()




#save figure 
plt.savefig('resultus_LSIFIR.PNG')
print("Training Results Saved.")
            
#%% Load Model if You Want 

loaded_model = Net()
loaded_model.load_state_dict(torch.load('trained_model.pth'))
print("Trained Model Loaded")
 