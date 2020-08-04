
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sklearn.metrics as metrics
import numpy as np


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

pdir, tdir = 'solar_pv/', 'testing/'
image_datasets = datasets.ImageFolder(pdir + 'training', data_transforms['train'])
#val_datasets = datasets.ImageFolder(pdir + 'val', data_transforms['val'])
test_datasets = datasets.ImageFolder(tdir, data_transforms['val'])

dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=30, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=30, shuffle=False, num_workers=4)
#valloader = torch.utils.data.DataLoader(val_datasets, batch_size=30, shuffle=False, num_workers=4)
#class_names = image_datasets.classes
#class_names = test_datasets.classes

print (test_datasets)
print (len(test_datasets))

#for i in range(len(test_datasets)):
#    print (test_datasets.samples[i])

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
#print (model_ft)

model_ft = model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

num_epoch = 20
#num_epoch = 0

for epoch in range(num_epoch):
    pred_l = []
    y_l = []
    for data in dataloader:
        model_ft.train()
        x, y = data
        x = Variable(x).cuda()
        y = Variable(y).cuda()
        pred = model_ft(x)

        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        _, pred_result = torch.max(pred, 1)
        pred_l.append(pred[:, -1].data.cpu().numpy() )
        y_l.append(y.data.cpu().numpy() )
        #print (y.data.cpu().numpy().shape)
        #print (pred[:, -1].data.cpu().numpy().shape)
    pred_l = np.concatenate(pred_l)
    y_l = np.concatenate(y_l)
    print (epoch, 'train', round(metrics.roc_auc_score(y_l, pred_l), 2))
        
 #   pred_l = []
 #   y_l = []
 #   for data in valloader:
 #       model_ft.eval()
 #       x, y = data
 #       x = Variable(x).cuda()
 #       y = Variable(y).cuda()
 #       pred = model_ft(x)

 #       pred_l.append(pred[:, -1].data.cpu().numpy())
 #       y_l.append(y.data.cpu().numpy())
 #   pred_l = np.concatenate(pred_l)
 #   y_l = np.concatenate(y_l)
 #   print (epoch, 'val', round(metrics.roc_auc_score(y_l, pred_l), 2))


pred_result_l = []
for data in testloader:
    model_ft.eval()
    x, y = data
    x = Variable(x).cuda()
    y = Variable(y).cuda()
    pred = model_ft(x)

    _, pred_result = torch.max(pred, 1)
    #print ('test', pred_result)
    #print (pred[:, -1].data.cpu().numpy())
    pred_result_l.append(pred[:, -1].data.cpu().numpy())
    print ()
pred_result_l = np.concatenate(pred_result_l)
print ('pred_result_l cnn', pred_result_l.shape)
np.save('pred_result_l', pred_result_l)


