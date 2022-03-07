import random
import os
import numpy as np
import torch
from  torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split


def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1


def load_data(opt):
    train_dataset=None
    test_dataset=None
    val_dataset=None
    test_N_dataset=None
    test_S_dataset = None
    test_V_dataset = None
    test_F_dataset = None
    test_Q_dataset = None

    if opt.dataset=="ecg":

        N_samples=np.load(os.path.join(opt.dataroot, "N_samples.npy")) #NxCxL
        S_samples=np.load(os.path.join(opt.dataroot, "S_samples.npy"))
        V_samples = np.load(os.path.join(opt.dataroot, "V_samples.npy"))
        F_samples = np.load(os.path.join(opt.dataroot, "F_samples.npy"))
        Q_samples = np.load(os.path.join(opt.dataroot, "Q_samples.npy"))

        # normalize all
        for i in range(N_samples.shape[0]):
            for j in range(opt.nc):
                N_samples[i][j]=normalize(N_samples[i][j][:])
        N_samples=N_samples[:,:opt.nc,:]

        for i in range(S_samples.shape[0]):
            for j in range(opt.nc):
                S_samples[i][j] = normalize(S_samples[i][j][:])
        S_samples = S_samples[:, :opt.nc, :]

        for i in range(V_samples.shape[0]):
            for j in range(opt.nc):
                V_samples[i][j] = normalize(V_samples[i][j][:])
        V_samples = V_samples[:, :opt.nc, :]

        for i in range(F_samples.shape[0]):
            for j in range(opt.nc):
                F_samples[i][j] = normalize(F_samples[i][j][:])
        F_samples = F_samples[:, :opt.nc, :]

        for i in range(Q_samples.shape[0]):
            for j in range(opt.nc):
                Q_samples[i][j] = normalize(Q_samples[i][j][:])
        Q_samples = Q_samples[:, :opt.nc, :]


        # train / test
        test_N,test_N_y, train_N,train_N_y = getFloderK(N_samples,opt.folder,0)
        # test_S,test_S_y, train_S,train_S_y = getFloderK(S_samples, opt.folder,1)
        # test_V,test_V_y, train_V,train_V_y = getFloderK(V_samples, opt.folder,1)
        # test_F,test_F_y, train_F,train_F_y = getFloderK(F_samples, opt.folder,1)
        # test_Q,test_Q_y, train_Q,train_Q_y = getFloderK(Q_samples, opt.folder,1)
        test_S, test_S_y = S_samples, np.ones((S_samples.shape[0], 1))
        test_V, test_V_y = V_samples, np.ones((V_samples.shape[0], 1))
        test_F, test_F_y = F_samples, np.ones((F_samples.shape[0], 1))
        test_Q, test_Q_y = Q_samples, np.ones((Q_samples.shape[0], 1))


        # train / val
        train_N, val_N, train_N_y, val_N_y = getPercent(train_N, train_N_y, 0.1, 0)
        test_S, val_S, test_S_y, val_S_y = getPercent(test_S, test_S_y, 0.1, 0)
        test_V, val_V, test_V_y, val_V_y = getPercent(test_V, test_V_y, 0.1, 0)
        test_F, val_F, test_F_y, val_F_y = getPercent(test_F, test_F_y, 0.1, 0)
        test_Q, val_Q, test_Q_y, val_Q_y = getPercent(test_Q, test_Q_y, 0.1, 0)

        val_data=np.concatenate([val_N,val_S,val_V,val_F,val_Q])
        val_y=np.concatenate([val_N_y,val_S_y,val_V_y,val_F_y,val_Q_y])


        test_data=np.concatenate([test_N,test_S,test_V,test_F,test_Q])
        test_y=np.concatenate([test_N_y,test_S_y,test_V_y,test_F_y,test_Q_y])

        print("train data size:{}".format(train_N.shape))
        print("val data size:{}".format(val_data.shape))
        print("test N data size:{}".format(test_N.shape))
        print("test S data size:{}".format(test_S.shape))
        print("test V data size:{}".format(test_V.shape))
        print("test F data size:{}".format(test_F.shape))
        print("test Q data size:{}".format(test_Q.shape))


        if not opt.istest and opt.n_aug>0:
            train_N, train_N_y = aug_noise(train_N,train_N_y,opt.n_aug)
            print("after aug, train data size:{}".format(train_N.shape))


        train_dataset = TensorDataset(torch.Tensor(train_N),torch.Tensor(train_N_y))
        val_dataset= TensorDataset(torch.Tensor(val_data), torch.Tensor(val_y))
        test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_y))
        # test_N_dataset = TensorDataset(torch.Tensor(test_N), torch.Tensor(test_N_y))
        # test_S_dataset = TensorDataset(torch.Tensor(test_S), torch.Tensor(test_S_y))
        # test_V_dataset = TensorDataset(torch.Tensor(test_V), torch.Tensor(test_V_y))
        # test_F_dataset = TensorDataset(torch.Tensor(test_F), torch.Tensor(test_F_y))
        # test_Q_dataset = TensorDataset(torch.Tensor(test_Q), torch.Tensor(test_Q_y))


    # assert (train_dataset is not None and test_dataset is not None and val_dataset is not None)
    dataloader = {"train": DataLoader(
                        dataset=train_dataset,  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=True),
                    "val": DataLoader(
                        dataset=val_dataset,  # torch TensorDataset format
                        batch_size=opt.batchsize,  # mini batch size
                        shuffle=True,
                        num_workers=int(opt.workers),
                        drop_last=False),
                    "test":DataLoader(
                            dataset=test_dataset,  # torch TensorDataset format
                            batch_size=opt.batchsize,  # mini batch size
                            shuffle=False,
                            num_workers=int(opt.workers),
                            drop_last=False),
                    # "test_S": DataLoader(
                    #     dataset=test_S_dataset,  # torch TensorDataset format
                    #     batch_size=opt.batchsize,  # mini batch size
                    #     shuffle=True,
                    #     num_workers=int(opt.workers),
                    #     drop_last=False),
                    # "test_V": DataLoader(
                    #     dataset=test_V_dataset,  # torch TensorDataset format
                    #     batch_size=opt.batchsize,  # mini batch size
                    #     shuffle=True,
                    #     num_workers=int(opt.workers),
                    #     drop_last=False),
                    # "test_F": DataLoader(
                    #     dataset=test_F_dataset,  # torch TensorDataset format
                    #     batch_size=opt.batchsize,  # mini batch size
                    #     shuffle=True,
                    #     num_workers=int(opt.workers),
                    #     drop_last=False),
                    # "test_Q": DataLoader(
                    #     dataset=test_Q_dataset,  # torch TensorDataset format
                    #     batch_size=opt.batchsize,  # mini batch size
                    #     shuffle=True,
                    #     num_workers=int(opt.workers),
                    #     drop_last=False),
                    }
    return dataloader


def getFloderK(data,folder,label):
    normal_cnt = data.shape[0]
    folder_num = int(normal_cnt / 5)
    folder_idx = folder * folder_num

    folder_data = data[folder_idx:folder_idx + folder_num]

    remain_data = np.concatenate([data[:folder_idx], data[folder_idx + folder_num:]])
    if label==0:
        folder_data_y = np.zeros((folder_data.shape[0], 1))
        remain_data_y=np.zeros((remain_data.shape[0], 1))
    elif label==1:
        folder_data_y = np.ones((folder_data.shape[0], 1))
        remain_data_y = np.ones((remain_data.shape[0], 1))
    else:
        raise Exception("label should be 0 or 1, get:{}".format(label))
    return folder_data,folder_data_y,remain_data,remain_data_y


def getPercent(data_x,data_y,percent,seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,test_size=percent,random_state=seed)
    return train_x, test_x, train_y, test_y


def aug_noise(train_x, train_y, times):
    res_train_x=[]
    res_train_y=[]
    for idx in range(train_x.shape[0]):
        # original
        x=train_x[idx]
        y=train_y[idx]
        res_train_x.append(x)
        res_train_y.append(y)

        for i in range(times):

            # augmented
            sigma = random.uniform(0.01,0.03)
            noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
            output = normalize(x+noise)

            res_train_x.append(output)
            res_train_y.append(y)

    res_train_x = np.array(res_train_x)
    res_train_y = np.array(res_train_y)
    return res_train_x,res_train_y