import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as model
import pandas as pd
import numpy as np
import time
from collections import OrderedDict
from torchvision import transforms, datasets
from tqdm import tqdm

from my_dataset import MyDataSet, MyTestDataSet
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

    data_root = cardiac_data_root
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)
    nfold = 10
    epochs = 100
    batch_size = 64
    learning_rate = 1e-5
    step1 = 'R10'
    step2 = 'R10'
    logname = 'PEAD-' + step1
    logname2 = 'PEAD-' + step1 + '-' + step2
    nonHeartname = step1 + '_nonHeart.csv'
    log_path = os.path.join(os.getcwd(), "logs", logname) + ".csv"
    save_root = os.path.join(project_root,'CODE','classification',"output")
    if not os.path.exists(save_root):
      os.makedirs(save_root)
      print("output save path created!")
    out_path = os.path.join(save_root, logname) + ".csv"
    # generate 10 fold for 10-fold cross-validation

    classes = os.listdir(data_root)
    non_cases_list = []
    idx_path = os.path.join(project_root, 'CODE', 'classification', 'case_order.txt')
    f = open(idx_path, "r")
    for line in f.readlines():
        non_cases_list.append(line[:-1])
    f.close()

    # record the metrics for each epoch
    # initialize the experiment log
    log = OrderedDict([
        ('fold', []),
        ('val size', []),
        ('val time', []),
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
    all_img_path = []

    for fold in range(0, nfold):

        # create dataset
        valid_cases = non_cases_list[fold * (nfold - 1):(fold + 1) * (nfold - 1)]  # get all cases in test set

        # initialize parameters
        valid_path = []
        valid_label = []

        for test_case in valid_cases:
            for cls in range(0, len(classes)):
                case_root = os.path.join(data_root, classes[cls], test_case)
                for img in os.listdir(case_root):
                    img_path = os.path.join(case_root, img)
                    valid_path.append(img_path)
                    valid_label.append(cls)

        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))


        valid_dataset = MyTestDataSet(images_path=valid_path,
                                  images_class=valid_label,
                                  transform=data_transform["val"])
        val_num = len(valid_dataset)
        foldnum = [fold+1]*val_num
        fold_num.extend(np.array(foldnum))
        validate_loader = torch.utils.data.DataLoader(valid_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw)


        net = resnet10(num_classes = 2)
        # load pretrain weights
        model_weight_name = "H-ResNet10-1_" + str(fold+1) + ".pth"
        model_weight_path = os.path.join(os.getcwd(), "save_weight", model_weight_name)
        # download url: https://download.pytorch.org/models/vgg13-c768596a.pth
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path, map_location=lambda storage, loc: storage),strict=False)
        # change fc layer structure
        # in_channel = net.classifier[-1].in_features
        # net.classifier[-1] = nn.Linear(in_channel, 2)
        net.to(device)

        # define loss function
        loss_function = nn.CrossEntropyLoss()

        # construct an optimizer
        params = [p for p in net.parameters() if p.requires_grad]
        # learning rate scheduler
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                                step_size=10,
        #                                                gamma=0.33)



        best_acc = 0.0

        fold_out = []
        fold_lab = []




        # validate
        net.eval()
        TP = 0.0
        TN = 0.0
        FN = 0.0
        FP = 0.0
        acc = 0.0  # accumulate accurate number / epoch

        score_list = []
        label_list = []
        GT_paths = []
        val_start_time = time.time()
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels, GT_path = val_data
                outputs = net(val_images.to(device))
                prob_tmp = torch.nn.Softmax(dim=1)(outputs)
                score_list.extend(prob_tmp.cpu().numpy())
                label_list.extend(val_labels.cpu().numpy())
                GT_paths.extend(GT_path)
                loss = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                TP += ((predict_y == 1) & (val_labels.to(device) == 1)).cpu().sum()
                TN += ((predict_y == 0) & (val_labels.to(device) == 0)).cpu().sum()
                FN += ((predict_y == 0) & (val_labels.to(device) == 1)).cpu().sum()
                FP += ((predict_y == 1) & (val_labels.to(device) == 0)).cpu().sum()

                val_bar.desc = "valid loss:{:.4f} acc:{:.4f}" \
                    .format(loss, acc)

        val_end_time = time.time()
        val_time = val_end_time - val_start_time
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)

        val_accurate = acc / val_num

        log['fold'].append(fold + 1)
        log['val size'].append(val_num)
        log['val time'].append(val_time)
        log['acc'].append(val_accurate)
        log['precision'].append(p.item())
        log['recall'].append(r.item())
        log['F1'].append(F1.item())
        log['TP'].append(TP.item())
        log['TN'].append(TN.item())
        log['FP'].append(FP.item())
        log['FN'].append(FN.item())

        pd.DataFrame(log).to_csv(log_path, index=False)
        print(
            'val_accuracy: %.4f test_loss: %.4f precision: %.4f recall: %.4f F1: %.4f' %
            (val_accurate, loss, p, r, F1))

        if val_accurate > best_acc:
            fold_out = score_list
            label_tensor = torch.tensor(label_list)
            label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
            label_onehot = torch.zeros(label_tensor.shape[0], 2)
            label_onehot.scatter_(dim=1, index=label_tensor, value=1)
            label_onehot = np.array(label_onehot)
            fold_lab = label_onehot

        out.extend(fold_out)
        lab.extend(fold_lab)
        all_img_path.extend(GT_paths)

    outs = np.array(out)
    labs = np.array(lab)
    data_list = [outs[:, 0], outs[:, 1], labs[:, 0], labs[:, 1], fold_num,all_img_path]
    data = pd.DataFrame(data_list).T
    print(data)
    data.to_csv(out_path, index=False)

########################################################################################################################

    log_path2 = os.path.join(os.getcwd(), "logs", logname2) + ".csv"
    save_root = os.path.join(project_root,'CODE','classification',"output")
    model_weight_path = os.path.join(os.getcwd(),"pretrain_weight", "resnet10-pre.pth")
    if not os.path.exists(save_root):
      os.makedirs(save_root)
      print("output save path created!")
    out_path2 = os.path.join(save_root, logname2) + ".csv"
    pred_nonH = OrderedDict([
        ("pred0",[]),
        ("pred1",[]),
        ("label0",[]),
        ("label1",[]),
        ("fold",[]),
        ("path",[])
    ])
    nonH_save = os.path.join(project_root,'CODE','classification',"output",nonHeartname)

    out = []
    lab = []
    fold_num = []
    all_img_path = []

    # initialize the experiment log
    log2 = OrderedDict([
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

    for fold in range(1,11):
        test_path = []
        test_label = []
        train_path = []
        train_label = []

        for row in data.itertuples():
            # Extract each fold and generate test set
            if getattr(row, '_5') == fold:
                if getattr(row, '_2') > getattr(row, '_1'):
                    path = str(getattr(row, '_6'))
                    splits = path.split("/")
                    image = splits[-1]
                    case = splits[-2]
                    cls = splits[-3]
                    if cls == "0Non_Heart":
                        test_label.append(1)
                        test_path.append(os.path.join(project_root,"Data", "NMDID","NMDID_HeartData","0Non_Heart",case,image))
                    else:
                        direction = os.path.join(project_root,"Data", "NMDID","PEwholedata","1PE",case,image)
                        if not os.path.exists(direction):
                            direction = os.path.join(project_root,"Data", "NMDID","PEwholedata","0nonPE",case,image)
                            test_label.append(1)
                            test_path.append(direction)
                        else:
                            test_label.append(0)
                            test_path.append(direction)
                else:
                    path = str(getattr(row, '_6'))
                    splits = path.split("/")
                    image = splits[-1]
                    case = splits[-2]
                    direction = os.path.join(project_root, "Data", "NMDID", "PEwholedata", "1PE", case, image)
                    if os.path.exists(direction):
                        pred_nonH["pred0"].append(0)
                        pred_nonH["pred1"].append(1)
                        pred_nonH["label0"].append(1)
                        pred_nonH["label1"].append(0)
                        pred_nonH["fold"].append(getattr(row, '_5'))
                        pred_nonH["path"].append(direction)
                    else:
                        pred_nonH["pred0"].append(0)
                        pred_nonH["pred1"].append(1)
                        pred_nonH["label0"].append(0)
                        pred_nonH["label1"].append(1)
                        pred_nonH["fold"].append(getattr(row, '_5'))
                        pred_nonH["path"].append(path)
            else:
                if getattr(row, '_2') > getattr(row, '_1'):
                    path = str(getattr(row, '_6'))
                    splits = path.split("/")
                    image = splits[-1]
                    case = splits[-2]
                    cls = splits[-3]
                    if cls != "0Non_Heart":
                        direction = os.path.join(project_root, "Data", "NMDID", "PEwholedata", "1PE", case, image)
                        if not os.path.exists(direction):
                            direction = os.path.join(project_root, "Data", "NMDID", "PEwholedata", "0nonPE", case,
                                                     image)
                            train_label.append(1)
                            train_path.append(direction)
                        else:
                            train_label.append(0)
                            train_path.append(direction)


        for i in range(len(train_label)):
            path = train_path[i]
            label = train_label[i]
            splits = path.split("/")
            image = splits[-1]
            case = splits[-2]
            cls = splits[-3]
            if cls == "0Non_Heart":
                if label != 1:
                    print(path, "is incorrect labeled!")
            elif cls == "1PE":
                if label != 0:
                    print(path, "is incorrect labeled!")
            elif cls == "0nonPE":
                if label != 1:
                    print(path, "is incorrect labeled!")
            else:
                print("path is unlabeled!",path)

        if fold == 1:
            c = {'label':train_label, "path":train_path}
            pd.DataFrame(c).to_csv('PEADtrain1.csv', index=False)
            d = {'label':test_label, "path":test_path}
            pd.DataFrame(d).to_csv('PEADtest1.csv', index=False)
                
        pd.DataFrame(pred_nonH).to_csv(nonH_save, index=False)



        print("test for cropped model!")

        valid_dataset2 = MyTestDataSet(images_path=test_path,
                                      images_class=test_label,
                                      transform=data_transform["val"])
        val_num = len(valid_dataset2)
        foldnum = [fold] * val_num
        fold_num.extend(np.array(foldnum))

        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))

        validate_loader2 = torch.utils.data.DataLoader(valid_dataset2,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw)

        train_dataset2 = MyTestDataSet(images_path=train_path,
                                  images_class=train_label,
                                  transform=data_transform["train"])
        train_num = len(train_dataset2)

        train_loader = torch.utils.data.DataLoader(train_dataset2,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=nw)

        net = resnet10()
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path, map_location=lambda storage, loc: storage), strict=False)
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, 2)
        net.to(device)

        # define loss function
        loss_function = nn.CrossEntropyLoss()

        # construct an optimizer
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=learning_rate)


        best_f1 = 0.0

        save_path2 = os.path.join(os.getcwd(), "save_weight", logname2) + "_" + str(fold) + ".pth"
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
            for step, traindata in enumerate(train_bar):
                images, labels, GTpaths = traindata
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
                    .format(fold + 1, nfold, epoch + 1,epochs,a,loss,train_acc)
            train_end_time = time.time()
            train_time = train_end_time - train_start_time
            print(train_time)

            # validate
            net.eval()
            TP = 0.0
            TN = 0.0
            FN = 0.0
            FP = 0.0
            acc = 0.0  # accumulate accurate number / epoch

            score_list = []
            label_list = []
            GT_paths = []
            val_start_time = time.time()
            with torch.no_grad():
                val_bar = tqdm(validate_loader2)
                for val_data in val_bar:
                    val_images, val_labels, GT_path = val_data
                    outputs = net(val_images.to(device))
                    prob_tmp = torch.nn.Softmax(dim=1)(outputs)
                    score_list.extend(prob_tmp.cpu().numpy())
                    label_list.extend(val_labels.cpu().numpy())
                    GT_paths.extend(GT_path)
                    loss = loss_function(outputs, val_labels.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    TP += ((predict_y == 1) & (val_labels.to(device) == 1)).cpu().sum()
                    TN += ((predict_y == 0) & (val_labels.to(device) == 0)).cpu().sum()
                    FN += ((predict_y == 0) & (val_labels.to(device) == 1)).cpu().sum()
                    FP += ((predict_y == 1) & (val_labels.to(device) == 0)).cpu().sum()

                    val_bar.desc = "valid loss:{:.4f} acc:{:.4f}" \
                        .format(loss, acc)

            val_end_time = time.time()
            val_time = val_end_time - val_start_time
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            F1 = 2 * r * p / (r + p)

            val_accurate = acc / val_num

            log2['fold'].append(fold)
            log2['epoch'].append(epoch+1)
            log2['train size'].append(train_num)
            log2['train time'].append(train_time)
            log2['val size'].append(val_num)
            log2['val time'].append(val_time)
            log2['lr'].append(a)
            log2['loss'].append(running_loss / train_steps)
            log2['acc'].append(val_accurate)
            log2['precision'].append(p.item())
            log2['recall'].append(r.item())
            log2['F1'].append(F1.item())
            log2['TP'].append(TP.item())
            log2['TN'].append(TN.item())
            log2['FP'].append(FP.item())
            log2['FN'].append(FN.item())

            pd.DataFrame(log2).to_csv(log_path2, index=False)
            print(
                '[epoch %d] lr: %.6f train_loss: %.4f train_acc: %.4f val_accuracy: %.4f test_loss: %.4f precision: %.4f recall: %.4f F1: %.4f' %
                (epoch + 1, a, running_loss / train_steps, train_acc, val_accurate, loss, p, r, F1))

            if F1 > best_f1:
                best_f1 = F1
                torch.save(net.state_dict(), save_path2)
                fold_out = score_list
                label_tensor = torch.tensor(label_list)
                label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
                label_onehot = torch.zeros(label_tensor.shape[0], 2)
                label_onehot.scatter_(dim=1, index=label_tensor, value=1)
                label_onehot = np.array(label_onehot)
                fold_lab = label_onehot
        out.extend(fold_out)
        lab.extend(fold_lab)
        all_img_path.extend(GT_paths)

    outs = np.array(out)
    labs = np.array(lab)
    data_list = [outs[:, 0], outs[:, 1], labs[:, 0], labs[:, 1], fold_num, all_img_path]
    data2 = pd.DataFrame(data_list).T
    data2.columns = ["pred0","pred1","label0","label1","fold","path"]
    frames = [data2,pd.DataFrame(pred_nonH)]
    NOD_data = pd.concat(frames)
    NOD_data.to_csv(out_path2, index=False)

if __name__ == '__main__':
    main()
