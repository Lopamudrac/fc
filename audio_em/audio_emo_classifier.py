from hashlib import new
import numpy as np
import torch
from audio_feature import feature_extraction
import torch.optim as optim


training_loader,training_y_loader,validation_loader,validation_y_loader,test_loader,test_y_loader,device, optimizer,loss_fn,learning_rate,resnet_model=feature_extraction()
res_train_loss=[]
res_valid_loss=[]


# print("training loader",training_loader,"train y loader",training_y_loader,"val loader",validation_loader,
# "val_y_loader",validation_y_loader,"test y loader",test_loader,"test_y_loader",test_y_loader,"optimizer", optimizer,
# "loss",loss_fn,"resnet",resnet_model)

# def label_converter(y_dataloader):
#     train_labels=[]
#     for y_list in y_dataloader:
#         for y in y_list:
#             train_labels.append(emotion_dict[y])
#     return torch.utils.data.DataLoader(train_labels, batch_size=10, shuffle=False)
def lr_decay(optimizer, epoch):
  if epoch%10==0:
    new_lr = learning_rate / (10**(epoch//10))
    optimizer = optim.Adam(resnet_model.parameters(), lr=new_lr)
    print(f'Changed learning rate to {new_lr}')
  return optimizer

def train_data(model,epochs, loss_fn, train_loader, train_y_loader, valid_loader, valid_y_loader,train_losses,valid_losses):
    # train_labels= label_converter(train_y_loader)
    # valid_labels= label_converter(valid_y_loader)
    #print("label",train_labels, train_labels, valid_labels[0], valid_labels[1])
    for epoch in range(1,epochs+1):
        model.train()
        batch_losses=[]
        #if change_lr:
            #optimizer = change_lr(optimizer, epoch)
        # for i,y in enumerate(train_y_loader):
        #     print(y)
        mod_optimizer=lr_decay(optimizer,epoch)
        trace_y = []
        trace_yhat = []
        for i,(x_data,y_data) in enumerate(zip(train_loader,train_y_loader)):
            x_dt = x_data
            y_dt = y_data
            mod_optimizer.zero_grad()
            x = x_dt.to(device, dtype=torch.float32)
            y = y_dt.to(device, dtype=torch.long)
            #print("x",x)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())      
            mod_optimizer.step()
        train_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
        print(f'Epoch - {epoch} Train-Loss : {np.mean(train_losses[-1])} Train-Accuracy : {accuracy}')
        model.eval()
        batch_losses=[]
        trace_y = []
        trace_yhat = []
        for i,(x_val_data, y_val_data) in enumerate(zip(valid_loader,valid_y_loader)):
            x = x_val_data
            y = y_val_data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())      
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        #print("y",trace_y)
        #print("yhat",trace_yhat.argmax(axis=1))
        accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)
        print(f'Epoch - {epoch} Valid-Loss : {np.mean(valid_losses[-1])} Valid-Accuracy : {accuracy}')

train_data(resnet_model,100,loss_fn, training_loader, training_y_loader, 
validation_loader,validation_y_loader,res_train_loss,res_valid_loss)
