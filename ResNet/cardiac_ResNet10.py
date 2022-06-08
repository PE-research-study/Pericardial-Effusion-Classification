import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as model
import numpy as np
import pandas as pd
import time
import random
from random import shuffle
from collections import OrderedDict
from torchvision import transforms, datasets
from tqdm import tqdm

from my_dataset import MyDataSet
from ResNet_model import resnet10

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}

    # hyperparameters

    project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))  # get data root path
    cardiac_data_root = os.path.join(project_root, "Data", "NMDID","NMDID_HeartData")  # caridac data set path
    # model_weight_path = os.path.join(os.getcwd(),"pretrain_weight", "resnet18-pre.pth")

    data_root = cardiac_data_root
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)
    nfold = 10
    epochs = 100
    batch_size = 64
    learning_rate = 1e-5
    logname = 'H-ResNet10'
    log_path = os.path.join(os.getcwd(), "logs", logname) + ".csv"
    save_root = os.path.join(project_root,'CODE','classification',"output")
    if not os.path.exists(save_root):
      os.makedirs(save_root)
      print("output save path created!")
    out_path = os.path.join(save_root, logname) + ".csv"
    # generate 10 fold for 10-fold cross-validation

    classes = os.listdir(data_root)
    non_cases_list = []
    idx_path = os.path.join(project_root,'CODE','classification','case_order.txt')
    f = open(idx_path, "r")
    for line in f.readlines():
        non_cases_list.append(line[:-1])
    f.close()


    # record the metrics for each epoch
    # initialize the experiment log
    log = OrderedDict([
        ('fold', []),
        ('epoch', []),
        ('train size', []),
        ('train time', []),
        ('val size', []),
        ('val time', []),
        ('lr', []),
        ('loss', []),
        ('acc', []),
        ('precision', []),
        ('recall', []),
        ('F1', []),
        ('TP', []),
        ('TN', []),
        ('FP', []),
        ('FN', []),
    ])

    out = []
    lab = []
    fold_num = []

    for fold in range(0,nfold):

        # create dataset
        valid_cases = non_cases_list[fold*(nfold-1):(fold+1)*(nfold-1)] # get all cases in test set
        train_cases = non_cases_list[:fold*(nfold-1)] + non_cases_list[(fold+1)*(nfold-1):] # get all cases in train set

        # initialize parameters
        train_path = []
        valid_path = []
        train_label = []
        valid_label = []
        for train_case in train_cases:
            for cls in range(0,len(classes)):
                case_root = os.path.join(data_root,classes[cls],train_case)
                for img in os.listdir(case_root):
                    img_path = os.path.join(case_root,img)
                    train_path.append(img_path)
                    train_label.append(cls)

        for test_case in valid_cases:
            for cls in range(0, len(classes)):
                case_root = os.path.join(data_root, classes[cls], test_case)
                for img in os.listdir(case_root):
                    img_path = os.path.join(case_root, img)
                    valid_path.append(img_path)
                    valid_label.append(cls)


        train_dataset = MyDataSet(images_path=train_path,
                               images_class=train_label,
                               transform=data_transform["train"])
        train_num = len(train_dataset)

        # # {'0Nonheart':0, '1Heart':1}
        # flower_list = train_dataset.class_to_idx
        # cla_dict = dict((val, key) for key, val in flower_list.items())
        # # write dict into json file
        # json_str = json.dumps(cla_dict, indent=2)
        # with open('class_indices.json', 'w') as json_file:
        #     json_file.write(json_str)

        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=nw)

        valid_dataset = MyDataSet(images_path=valid_path,
                                  images_class=valid_label,
                                  transform=data_transform["val"])
        val_num = len(valid_dataset)
        foldnum = [fold+1]*val_num
        fold_num.extend(np.array(foldnum))
        validate_loader = torch.utils.data.DataLoader(valid_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw)

        print("using {} images for training, {} images for validation.".format(train_num,
                                                                               val_num))


        net = resnet10()
        # load pretrain weights
        # download url: https://download.pytorch.org/models/vgg13-c768596a.pth
        # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        # net.load_state_dict(torch.load(model_weight_path, map_location=lambda storage, loc: storage),strict=False)
        # for param in net.parameters():
        #     param.requires_grad = False
        #
        # change fc layer structure
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, 2)
        net.to(device)

        # define loss function
        loss_function = nn.CrossEntropyLoss()

        # construct an optimizer
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=learning_rate)
        # learning rate scheduler
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                                step_size=10,
        #                                                gamma=0.33)


        best_acc = 0.0

        save_fold = fold + 1
        save_path = os.path.join(os.getcwd(),"save_weight",logname) + "_" + str(save_fold) + ".pth"
        train_steps = len(train_loader)

        fold_out = []
        fold_lab = []

        for epoch in range(epochs):

            # train
            net.train()
            running_loss = 0.0
            epoch_T = 0.0
            train_acc = 0.0
            train_bar = tqdm(train_loader)
            train_start_time = time.time()
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                predict_y = torch.max(logits, dim=1)[1]
                epoch_T += torch.eq(predict_y, labels.to(device)).sum().item()
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()
                train_acc = epoch_T / (batch_size * (step + 1))
                a = optimizer.state_dict()['param_groups'][0]['lr']
                train_bar.desc = "train fold[{}/{}] train epoch[{}/{}] lr:{:.6f} loss:{:.3f} train acc:{:.4f}"\
                    .format(fold + 1, nfold, epoch + 1,a,epochs,loss,train_acc)
            train_end_time = time.time()
            train_time = train_end_time - train_start_time
            print(train_time)
            # update the learning rate
            # lr_scheduler.step()

            # validate
            net.eval()
            TP = 0.0
            TN = 0.0
            FN = 0.0
            FP = 0.0
            acc = 0.0  # accumulate accurate number / epoch

            score_list = []
            label_list = []
            val_start_time = time.time()
            with torch.no_grad():
                val_bar = tqdm(validate_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    prob_tmp = torch.nn.Softmax(dim=1)(outputs)
                    score_list.extend(prob_tmp.cpu().numpy())
                    label_list.extend(val_labels.cpu().numpy())
                    loss = loss_function(outputs, val_labels.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    TP += ((predict_y == 1)&(val_labels.to(device) == 1)).cpu().sum()
                    TN += ((predict_y == 0)&(val_labels.to(device) == 0)).cpu().sum()
                    FN += ((predict_y == 0)&(val_labels.to(device) == 1)).cpu().sum()
                    FP += ((predict_y == 1)&(val_labels.to(device) == 0)).cpu().sum()

                    val_bar.desc = "valid epoch[{}/{}] loss:{:.4f} acc:{:.4f}"\
                        .format(epoch + 1, epochs, loss, acc)

            val_end_time = time.time()
            val_time = val_end_time - val_start_time
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            F1 = 2 * r * p / (r + p)

            val_accurate = acc / val_num

            log['fold'].append(fold+1)
            log['epoch'].append(epoch+1)
            log['train size'].append(train_num)
            log['train time'].append(train_time)
            log['val size'].append(val_num)
            log['val time'].append(val_time)
            log['lr'].append(a)
            log['loss'].append(running_loss / train_steps)
            log['acc'].append(val_accurate)
            log['precision'].append(p.item())
            log['recall'].append(r.item())
            log['F1'].append(F1.item())
            log['TP'].append(TP.item())
            log['TN'].append(TN.item())
            log['FP'].append(FP.item())
            log['FN'].append(FN.item())

            pd.DataFrame(log).to_csv(log_path, index=False)
            print('[epoch %d] lr: %.6f train_loss: %.4f train_acc: %.4f val_accuracy: %.4f test_loss: %.4f precision: %.4f recall: %.4f F1: %.4f' %
                  (epoch + 1, a, running_loss / train_steps, train_acc, val_accurate, loss, p,r,F1))

            if F1 > best_acc:
                best_acc = F1
                torch.save(net.state_dict(), save_path)
                fold_out = score_list
                label_tensor = torch.tensor(label_list)
                label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
                label_onehot = torch.zeros(label_tensor.shape[0], 2)
                label_onehot.scatter_(dim=1, index=label_tensor, value=1)
                label_onehot = np.array(label_onehot)
                fold_lab = label_onehot
        out.extend(fold_out)
        lab.extend(fold_lab)
    outs = np.array(out)
    labs = np.array(lab)
    data_list = [outs[:,0],outs[:,1],labs[:,0],labs[:,1],fold_num]
    data = pd.DataFrame(data_list).T
    print(data)
    data.to_csv(out_path,index=False)




if __name__ == '__main__':
    main()
