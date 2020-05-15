import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



df = pd.read_csv("countyRisks.csv")
death_rate = df.deaths/df.cases                                                                                         
df.insert(68,"death_rate", death_rate)                                                                                       


df = df.where(df.cases > 100)  
df = df.dropna()
d = df.death_rate.where(df.death_rate >= 0.038454672046274146, 0)                                                     
d = d.where(d == 0, 1) 
df.insert(69, "at_risk", d)
df.index = range(647)
train, val = train_test_split(df, test_size=0.2)


feats = ['percent_fair_or_poor_health',
       'average_number_of_physically_unhealthy_days', 
       'percent_smokers','percent_adults_with_obesity', 'percent_physically_inactive',
       'percent_excessive_drinking', 'population',
       'percent_some_college', 'num_associations', 'social_association_rate',
       'percent_severe_housing_problems', 'percent_frequent_physical_distress', 
       'percent_adults_with_diabetes','percent_food_insecure',
       'percent_insufficient_sleep','percent_homeowners', 'percent_less_than_18_years_of_age',
       'percent_65_and_over']



train_x = train[feats]
val_x = val[feats] 

x = val_x.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
val_x = pd.DataFrame(x_scaled)

t = train_x.values #returns a numpy array
new_min_max_scaler = preprocessing.MinMaxScaler()
t_scaled = new_min_max_scaler.fit_transform(t)
train_x = pd.DataFrame(t_scaled)

train_x = torch.FloatTensor(train_x.values)

val_x = torch.FloatTensor(val_x.values)                                                                                                                                  

train_y = list(train['at_risk'])
val_y = list(val['at_risk']) 

def vectorize(column):
    for i in range(len(column)):
        if column[i] == 0.0:
            column[i] = [1.0,0.0]
        elif column[i] == 1.0:
            column[i] = [0.0, 1.0]    
vectorize(train_y)
vectorize(val_y)

train_y = torch.FloatTensor(train_y)  
val_y = torch.FloatTensor(val_y) 




class MyFirstNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
   
        super(MyFirstNetwork, self).__init__()
        
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
      
        layer1_output = self.layer1(x)
        
        h = F.relu(layer1_output)
      
        logits = self.layer2(h)
        return F.softmax(logits, dim=1)

class MySecondNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        #Inherit the nn class
        super(MySecondNetwork, self).__init__()
    
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):

        layer1_output = self.layer1(x)
        h = F.relu(layer1_output)
        layer2_output = self.layer2(h)
        j = F.relu(layer2_output)
        logits = self.layer3(j)
        return F.softmax(logits, dim=1)

class MyFatNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        #Inherit the nn class
        super(MyFatNetwork, self).__init__()
    
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, hidden_size)
        self.layerOut= nn.Linear(hidden_size, output_size)
        
    def forward(self, x):

        layer1_output = self.layer1(x)
        h1 = F.relu(layer1_output)
        layer2_output = self.layer2(h1)
        h2 = F.relu(layer2_output)
        layer3_output = self.layer3(h1)
        h3 = F.relu(layer3_output)
        layer4_output = self.layer4(h1)
        h4 = F.relu(layer4_output)
        layer5_output = self.layer5(h4)
        h5 = F.relu(layer5_output)
        logits = self.layerOut(h5)
        return F.softmax(logits)

def train(network, data_matrix, labels, criterion, optimizer, num_epochs = 1000):
    losses = []
    for epoch in range(num_epochs):
        # Zeroing the gradient like before
        optimizer.zero_grad()
        # Getting the prediction from our model on all of the data
        output = network(data_matrix)
        # Calculating loss
        loss = criterion(output, labels)
        losses.append(loss)
        # Calculate the gradients at each step
        loss.backward()
        # Take a step in the appropriate direction
        optimizer.step()
        
    return losses

HS = 40
LR = .03
EPOCHS = 1
NET = 2



NETTYPE = [MyFirstNetwork,MySecondNetwork,MyFatNetwork][NET]

data = train_x
labels = train_y
data2 = val_x
labels2 = val_y

input_size = len(feats)

hidden_size = HS

output_size = 2

learning_rate = LR

mynet = NETTYPE(input_size, hidden_size, output_size)

# pytorch provides an mse calculator for us
criterion = nn.MSELoss()

optimizer = optim.SGD(mynet.parameters(), lr=learning_rate)

losses = train(mynet, data, labels, criterion, optimizer, EPOCHS)
losses2 = train(mynet, data2, labels2, criterion, optimizer, 1)

print(losses2[0].data.item())
print(losses[-1].data.item())

plt.plot(np.arange(len(losses)), losses)
plt.title("MSE Per Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

# fd= {}
# for size in range(20,25):

#     for learning_rate in [0.003, 0.006, 0.01, 0.03, 0.05, 0.1]:
    
#         mynet = NETTYPE(input_size, size, output_size)

#         # pytorch provides an mse calculator for us
#         criterion = nn.MSELoss()

#         optimizer = optim.SGD(mynet.parameters(), lr=learning_rate)

#         losses = train(mynet, data, labels, criterion, optimizer, EPOCHS)
#         losses2 = train(mynet, data2, labels2, criterion, optimizer, 1)

#         fd[f"Hidden Layer Size: {size}, Learning Rate: {learning_rate}"] =  losses2[0].data.item() - losses[-1].data.item()
        

# temp = min(fd.values()) 
# res = [key for key in fd if fd[key] == temp] 