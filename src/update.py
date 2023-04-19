import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int (i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self,item): # image와 label을 tensor 타입을 얻는다.
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs)
        ) # data split과정
        self.device = 'cuda' if args.gpu else 'cpu'
        # loss function은 NLL로 default
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int((5./7.)*len(idxs))]
        idxs_val = idxs[int((5./7.)*len(idxs)):int((6./7.)*len(idxs))]
        idxs_test = idxs[int((6./7.)*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), # int(len(idxs_train)) self.args.local_bs
                                 batch_size=int(len(idxs_train)), shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10),shuffle = False)

        return trainloader, validloader, testloader

    def update_weights(self, Client_model, Server_model, client_h, global_round):
        # train model
        Client_model.train()
        Server_model.train()
        client_h.train()
        epoch_loss=[]

        # optimizer 설정
        if self.args.optimizer == 'sgd':
            client_optimizer = torch.optim.SGD(Client_model.parameters(), lr=self.args.lr, momentum=0.5)
            #server_optimizer = torch.optim.SGD(Server_model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            client_optimizer = torch.optim.Adam(Client_model.parameters(),lr = self.args.lr, weight_decay = 1e-4)
            #server_optimizer = torch.optim.Adam(Server_model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss=[]
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # training
                Client_model.zero_grad()
                Server_model.zero_grad()
                client_h.zero_grad()

                client_output = Client_model(images)
                h_output = client_h(client_output)
                server_output = Server_model(client_output)

                c_loss = self.criterion(h_output, labels) # client-side loss
                s_loss = self.criterion(server_output, labels) # server-side loss
                loss = self.args.gamma * c_loss + (1-self.args.gamma) * s_loss # mult-exit loss
                loss.backward()
                client_optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx*len(images),
                        len(self.trainloader.dataset),
                        100.* batch_idx / len(self.trainloader), loss.item())
                    )
                self.logger.add_scalar('loss',loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return Server_model.state_dict(),Client_model.state_dict(),client_h.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model_c, model_h, model_s):
        """

        :param model:
        :return: inference accuracy and loss
        """

        model_c.eval()
        model_s.eval()
        model_h.eval()

        loss, total, correct = 0.0,0.0,0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            client_output = model_c(images)
            h_output = model_h(client_output)

            #entropy 계산
            outputs_ = []
            for num in range(images.shape[0]):
                entropy = 0
                for label_p in range(10):
                    output_numpy = h_output[num].detach().numpy()
                    entropy += - (output_numpy[label_p])*(np.log2(np.abs(output_numpy[label_p])))

                # 기준 엔트로피보다 작다면 client-side model
                if entropy < self.args.entropy:
                    outputs = h_output[num]
                    batch_loss = self.criterion(outputs, labels[num])
                    loss += batch_loss.item()

                    outputs = torch.unsqueeze(outputs, 0)
                    _, pred_labels = torch.max(outputs,1)
                    pred_labels = pred_labels.view(-1)
                    correct += torch.sum(torch.eq(pred_labels, labels[num])).item()

                    # outputs_list = outputs.detach().numpy()
                    # outputs_.append(outputs_list)


                # 기준 엔트로피보다 크다면 server-side model
                else :
                    outputs = model_s(torch.unsqueeze(client_output[num],0))
                    batch_loss = self.criterion(outputs[0], labels[num])
                    loss += batch_loss.item()

                    outputs = torch.unsqueeze(outputs, 0)
                    _, pred_labels = torch.max(outputs, 1)
                    pred_labels = pred_labels.view(-1)
                    correct += torch.sum(torch.eq(pred_labels, labels[num])).item()

                    # outputs_list = outputs.detach().numpy()
                    # outputs_.append(outputs_list)


            # Prediction
            # outputs_ = torch.from_numpy(np.array(outputs_))
            # outputs_ = torch.Tensor(outputs_)

            total += len(labels)

        accuracy = correct / total
        return accuracy, loss

def softmax(z):
    array_z= z - np.max(z)
    exp_x = np.array(array_z)
    result = exp_x/np.sum(exp_x)
    return result

def test_inference(args, model_c, model_h,model_s, test_dataset, idx):
    model_c.eval()
    model_s.eval()
    model_h.eval()

    loss, total, correct = 0.0,0.0,0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(DatasetSplit(test_dataset, idx),
                            batch_size=int(len(idx) / 10), shuffle=False) #d

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        client_output = model_c(images)
        h_output = model_h(client_output)

        # entropy 계산
        outputs_ = []
        for num in range(images.shape[0]):
            entropy = 0
            for label_p in range(10):
                output_numpy = h_output[num].detach().numpy()
                entropy += - (output_numpy[label_p]) * (np.log2(np.abs(output_numpy[label_p])))

            # 기준 엔트로피보다 작다면 client-side model
            if entropy < args.entropy:
                outputs = h_output[num]
                batch_loss = criterion(outputs, labels[num])
                loss += batch_loss.item()

                outputs = torch.unsqueeze(outputs, 0)
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels[num])).item()

                # outputs_list = outputs.detach().numpy()
                # outputs_.append(outputs_list)


            # 기준 엔트로피보다 크다면 server-side model
            else:
                outputs = model_s(torch.unsqueeze(client_output[num], 0))
                batch_loss = criterion(outputs[0], labels[num])
                loss += batch_loss.item()

                outputs = torch.unsqueeze(outputs, 0)
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels[num])).item()

                # outputs_list = outputs.detach().numpy()
                # outputs_.append(outputs_list)

        # Prediction
        # outputs_ = torch.from_numpy(np.array(outputs_))
        # outputs_ = torch.Tensor(outputs_)

        total += len(labels)

    accuracy = correct / total


    return accuracy, loss

