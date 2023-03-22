import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNMnist, CNNFMnist_client, CNNFMnist_server, h_classifier
from utils import get_dataset, average_weights, exp_details, c_aggregation,h_aggregation

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    start_time = time.time()

    #define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    #if args.gpu_id:
    #    torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    #load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    #Bulid Model ; client-side/server-side
    if args.model == 'cnn':
        client_model = CNNFMnist_client(args=args)
        server_model = CNNFMnist_server(args=args)
        client_h = h_classifier(args=args)

    else:
        exit('Error : unrecognized model')

    # set the model to train and send it to device
    client_model.to(device)
    client_model.train()
    print(client_model)

    client_h.to(device)
    client_h.train()
    print(client_h)

    server_model.to(device)
    server_model.train()
    print(server_model)

    # copy weights
    client_weight = client_model.state_dict()
    server_weight = server_model.state_dict()
    h_weight = client_h.state_dict()

    # training
    train_loss, train_accuracy = [],[]
    val_acc_list, net_list = [],[]
    cv_loss, cv_acc = [],[]
    print_every = 2
    val_loss_pre, counter = 0, 0


    # training / inference 구할 때 따로 따로 나눠서 하나 돌려놓고 하나 돌리는
    for epoch in tqdm(range(args.epochs)):
        server_weights, client_weights, client_h_weights, local_losses = [], [], [], []

        print(f'\n | Global Training Round : {epoch+1} |\n')

        client_model.train()
        client_h.train()
        server_model.train()

        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)

            if epoch != 0:
                #각 client에 global round 끝났을 때 update
                update_client_m = copy.deepcopy(client_model).load_state_dict(client_weight)
                update_client_h = copy.deepcopy(client_h).load_state_dict(h_weight)

            else :  #제일 첫 global round
                update_client_m = copy.deepcopy(client_model)
                update_client_h = copy.deepcopy(client_h)

            s_p, c_p, c_h, loss = local_model.update_weights(
                Client_model=update_client_m,Server_model=copy.deepcopy(server_model),
                client_h=update_client_h, global_round=epoch
            )

            # 각 client 에서 학습 시킨 파라미터 저장
            server_weights.append(copy.deepcopy(s_p))
            client_weights.append(copy.deepcopy(c_p))
            client_h_weights.append(copy.deepcopy(c_h))
            local_losses.append(copy.deepcopy(loss))

        # update server weights
        server_weight = average_weights(server_weights)
        server_model.load_state_dict(server_weight)

        # update client weights
        client_weight = c_aggregation(client_weights)
        h_weight = h_aggregation(client_h_weights)


        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

print(" ")
# Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

"""   
file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].npy'. \
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
           args.local_ep, args.local_bs)
np.save(file_name, train_accuracy)
"""