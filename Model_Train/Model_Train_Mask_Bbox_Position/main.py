import shutil
import pandas as pd
import datetime
import torch as t
import torch.optim as optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as transf
import numpy as np
import os
from model import Net
import torch
import torch.nn as nn
from datafeed import DataFeed
from sklearn.model_selection import train_test_split

data_path = "../../Data_Processing/scenario3.csv"  #choose which scenario to train
output_dir = "../../Data_Processing/"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_path)
train_df, temp_df = train_test_split(df, test_size=0.3)  #7：3
val_df, test_df = train_test_split(temp_df, test_size=0.333333333333334)  #2：1


train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)


all_cars = True
dayTime = datetime.datetime.now().strftime('%m-%d-%Y')
hourTime = datetime.datetime.now().strftime('%H_%M')
rootPath = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
modelsavedPath = rootPath + '//' + 'Model_Test' + '//' + dayTime + '_' + hourTime + '_' + "Model_Test_Mask_Bbox_Position"
filesPath = modelsavedPath + '//' + 'saved_analysis_files'
checkpointPath = modelsavedPath + '//' + 'checkpoint'
trainDataPath = rootPath + '//' + 'Data_Processing' + '//' + 'train.csv'
valDataPath = rootPath + '//' + 'Data_Processing' + '//' + 'val.csv'
isExists = os.path.exists(modelsavedPath)
if not isExists:
    os.makedirs(modelsavedPath)
isExists = os.path.exists(filesPath)
if not isExists:
    os.makedirs(filesPath)
isExists = os.path.exists(checkpointPath)
if not isExists:
    os.makedirs(checkpointPath)



# Hyper-parameters
batch_size = 32
val_batch_size = 1
lr = 0.01
image_grab = False
num_epochs = 100
train_size = [1]
img_resize = transf.Resize((32, 32))
proc_pipe = transf.Compose(
    [transf.ToPILImage(),
     img_resize,
     transf.ToTensor()]
)

train_loader = DataLoader(DataFeed(trainDataPath, trainDataPath, transform=proc_pipe),
                          batch_size=batch_size,
                          shuffle=True)
val_loader = DataLoader(DataFeed(valDataPath, valDataPath, transform=proc_pipe),
                        batch_size=val_batch_size,
                        shuffle=True)
now = datetime.datetime.now()
val_acc = []
top_1 = np.zeros((1, len(train_size)))
top_2 = np.zeros((1, len(train_size)))
top_3 = np.zeros((1, len(train_size)))
acc_loss = 0
itr = []
for idx, n in enumerate(train_size):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    opt = optimizer.Adam(net.parameters(), lr=lr)
    LR_sch = optimizer.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=15, min_lr=1e-3)
    count = 0
    running_loss = []
    running_top1_acc = []
    running_top2_acc = []
    running_top3_acc = []
    best_accuracy = 0
    for epoch in range(num_epochs):
        print('Epoch No. ' + str(epoch + 1))
        skipped_batches = 0
        print('Start training')
        for tr_count, (img, bbox_gps, label) in enumerate(train_loader):
            net.train()
            img = torch.where(img < 0.01, 0.0, 1.0)
            opt.zero_grad()
            label = label
            out = net.forward(img, bbox_gps)
            L = criterion(out, label)
            L.backward()
            opt.step()
            batch_loss = L.item()
            acc_loss += batch_loss
            count += 1
            if np.mod(count, 50) == 0:
                print('Training-Batch No.' + str(count))
                running_loss.append(batch_loss)  # running_loss.append()
                itr.append(count)
                print('Loss = ' + str(running_loss[-1]))
        print('Start validation')
        ave_top1_acc = 0
        ave_top2_acc = 0
        ave_top3_acc = 0
        ind_ten = t.as_tensor([0, 1, 2])
        top1_pred_out = []
        top2_pred_out = []
        top3_pred_out = []
        total_count = 0
        gt_beam = []
        for val_count, (imgs, bbox_gps, labels) in enumerate(val_loader):
            net.eval()
            imgs = torch.where(imgs < 0.01, 0.0, 1.0)  #Display standardization and remove small noise
            gt_beam.append(labels.detach().cpu().numpy()[0].tolist())
            opt.zero_grad()
            labels = labels
            total_count += labels.size(0)
            out = net.forward(imgs,bbox_gps)
            sorted_out = t.argsort(out, dim=1, descending=True)
            top_1_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:1])
            top1_pred_out.append(top_1_pred.cpu().numpy())
            top_2_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:2])
            top2_pred_out.append(top_2_pred.cpu().numpy())
            top_3_pred_0 = t.index_select(sorted_out, dim=1, index=ind_ten[0])
            top_3_pred_1 = t.index_select(sorted_out, dim=1, index=ind_ten[1])
            top_3_pred_2 = t.index_select(sorted_out, dim=1, index=ind_ten[2])
            top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten)
            tmp = [top_3_pred_0.item(), top_3_pred_1.item(), top_3_pred_2.item()]
            top3_pred_out.append(tmp)
            reshaped_labels = labels.reshape((labels.shape[0], 1))
            tiled_2_labels = reshaped_labels.repeat(1, 2)
            tiled_3_labels = reshaped_labels.repeat(1, 3)
            batch_top1_acc = t.sum(top_1_pred == labels, dtype=t.float32)
            batch_top2_acc = t.sum(top_2_pred == tiled_2_labels, dtype=t.float32)
            batch_top3_acc = t.sum(top_3_pred == tiled_3_labels, dtype=t.float32)
            ave_top1_acc += batch_top1_acc.item()
            ave_top2_acc += batch_top2_acc.item()
            ave_top3_acc += batch_top3_acc.item()
        running_top1_acc.append(ave_top1_acc / total_count)
        running_top2_acc.append(ave_top2_acc / total_count)
        running_top3_acc.append(ave_top3_acc / total_count)
        print('Average Top-1 accuracy {}'.format(running_top1_acc[-1]))
        print('Average Top-2 accuracy {}'.format(running_top2_acc[-1]))
        print('Average Top-3 accuracy {}'.format(running_top3_acc[-1]))
        print(running_top1_acc)
        print("Saving the predicted value in a csv file")
        cur_accuracy  = running_top1_acc[-1]
        print("Current acc:", cur_accuracy)
        print("Best acc:", best_accuracy)
        if cur_accuracy > best_accuracy:
            print("Saving the best model")
            net_name = checkpointPath  + '//' +  'Best_Model'
            model_name = checkpointPath  + '//' +  'Model.pth'
            t.onnx.export(net, (torch.randn(1, 1, 32, 32), torch.randn(1, 1, 6)), model_name)
            t.save(net.state_dict(), net_name)
            best_accuracy =  cur_accuracy
        print("Updated best acc", best_accuracy)
        print("Saving the predicted value in a csv file")
        file_to_save = f'{filesPath}//top1_pred_beam_val_after_{epoch+1}th_epoch.csv'
        indx = np.arange(1, len(top1_pred_out)+1, 1)
        df1 = pd.DataFrame()
        df1['index'] = indx
        df1['link_status'] = gt_beam
        df1['top1_pred'] = top1_pred_out
        df1['top2_pred'] = top2_pred_out
        df1['top3_pred'] = top3_pred_out
        df1.to_csv(file_to_save, index=False)
        LR_sch.step(L)
    top_1[0, idx] = running_top1_acc[-1]
    top_2[0, idx] = running_top2_acc[-1]
    top_3[0, idx] = running_top3_acc[-1]


    ################### Temp - Test the best model ################################
    print("\n================================================================\n")
    testDataPath = rootPath + '//' + 'Data_Processing' + '//' + 'test.csv'
    net.load_state_dict(torch.load(checkpointPath  + '//' +  'Best_Model'))
    net.eval()
    net = net
    test_loader = DataLoader(DataFeed(testDataPath,testDataPath, transform=proc_pipe),
                             batch_size=val_batch_size,
                             shuffle=False)
    print('Start Test')
    ave_top1_acc = 0
    ave_top2_acc = 0
    ave_top3_acc = 0
    ind_ten = t.as_tensor([0, 1, 2])
    top1_pred_out = []
    top2_pred_out = []
    top3_pred_out = []
    total_count = 0
    gt_beam = []
    for val_count, (imgs, bbox_gps, labels) in enumerate(test_loader):
        net.eval()
        imgs = torch.where(imgs < 0.01, 0.0, 1.0)
        gt_beam.append(labels.detach().cpu().numpy()[0].tolist())
        opt.zero_grad()
        labels = labels
        total_count += labels.size(0)
        out = net.forward(imgs, bbox_gps)
        sorted_out = t.argsort(out, dim=1, descending=True)
        top_1_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:1])
        top1_pred_out.append(top_1_pred.cpu().numpy())
        top_2_pred = t.index_select(sorted_out, dim=1, index=ind_ten[0:2])
        top2_pred_out.append(top_2_pred.cpu().numpy())
        top_3_pred_0 = t.index_select(sorted_out, dim=1, index=ind_ten[0])
        top_3_pred_1 = t.index_select(sorted_out, dim=1, index=ind_ten[1])
        top_3_pred_2 = t.index_select(sorted_out, dim=1, index=ind_ten[2])
        top_3_pred = t.index_select(sorted_out, dim=1, index=ind_ten)
        tmp = [top_3_pred_0.item(), top_3_pred_1.item(), top_3_pred_2.item()]
        top3_pred_out.append(tmp)
        reshaped_labels = labels.reshape((labels.shape[0], 1))
        tiled_2_labels = reshaped_labels.repeat(1, 2)
        tiled_3_labels = reshaped_labels.repeat(1, 3)
        batch_top1_acc = t.sum(top_1_pred == labels, dtype=t.float32)
        batch_top2_acc = t.sum(top_2_pred == tiled_2_labels, dtype=t.float32)
        batch_top3_acc = t.sum(top_3_pred == tiled_3_labels, dtype=t.float32)
        ave_top1_acc += batch_top1_acc.item()
        ave_top2_acc += batch_top2_acc.item()
        ave_top3_acc += batch_top3_acc.item()
    print("total test examples are", total_count)
    running_top1_acc.append(ave_top1_acc / total_count)  # (batch_size * (count_2 + 1)) )
    running_top2_acc.append(ave_top2_acc / total_count)
    running_top3_acc.append(ave_top3_acc / total_count)  # (batch_size * (count_2 + 1)))
    print('Average Top-1 accuracy {}'.format(running_top1_acc[-1]))
    print('Average Top-2 accuracy {}'.format(running_top2_acc[-1]))
    print('Average Top-3 accuracy {}'.format(running_top3_acc[-1]))