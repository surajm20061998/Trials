#LSTM implementation
#RNNs using Pytorch's implementation

import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 


#device config
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper parameters
#input_size = 784 #images are 28x28
input_size = 28
sequence_length = 28
num_layers = 2

hidden_size = 500 #can try out more
num_classes = 10
num_epochs = 10
batch_size = 100
leraning_rate = 0.001

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,),(0.3081,))])
#import MNIST data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                          transform = transform, download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                         transform = transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape)
print(labels.shape)


#Graphing some numbers

for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #x -> batch_size*seq*input_size
        self.fc = nn.Linear(hidden_size, num_classes)
        
        
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out,_ = self.lstm(x,(h0,c0))
        #batch_size, seq_length, hidden_size
        # out (N, 28, 128)
        out = out[:,-1,:]
        #out (N, 128)
        out = self.fc(out)
        return out
        
    
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)    

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=leraning_rate)

step_losses = []   # loss per mini-batch step
epoch_losses = []  # average loss per epoch

#Trainig loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    running_loss = 0.0
    for i , (images, labels) in enumerate(train_loader):
        #reshape images first
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        #backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        step_losses.append(loss.item())
        running_loss += loss.item()
        
        if (i+1) % 100 == 0:
            print(f'epoch {epoch +1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
    
    avg_epoch_loss = running_loss/n_total_steps
    epoch_losses.append(avg_epoch_loss)
    print(f'epoch {epoch+1} average loss: {avg_epoch_loss:.4f}')
    
            
#if I wanted a graph here showing me the loss curve for every step how would I do it?
plt.figure()
plt.plot(step_losses)
plt.xlabel("Step (mini-batch)")
plt.ylabel("Loss")
plt.title("Training Loss per Step")
plt.show()

#if I wanted a graph here showing me the loss curve for every epoch how would I do it?
plt.figure()
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Training Loss per Epoch")
plt.xticks(range(1, num_epochs + 1))
plt.show()
            

#Testing
with torch.no_grad():
    n_correct=0
    n_samples=0
    
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        
    acc = 100.0 * n_correct/n_samples
    print(f'accuracy = {acc}')
    
    
    torch.save(model.state_dict(), "mnist_ffn.pth")