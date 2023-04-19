import numpy as np
from torchvision import datasets, transforms
import random
import tensorflow as tf

def mnist_iid(dataset, num_users):
    """

    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {},[i for i in range(len(dataset))]
    for i in range(num_users): #모든 인덱스에서 item의 수만큼의 개수를 추출 (set으로 묶어주면서 중복제거)
        dict_users[i] = set(np.random.choice(all_idxs,num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i]) #유저의 추출데이터를 중복되지 않도록 set에서 빼준다.
    return dict_users

def mnist_noniid(dataset, num_users):
    # set random seed
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    dict_users = {i: np.array([]) for i in range(num_users)}
    y_train = dataset.train_labels

    # 랜덤으로 두 개의 클래스 선택
    class_labels = np.random.choice(np.unique(dataset.train_labels), size=100, replace=True).reshape(num_users, 2)

    # 각 클래스(또는 레이블)의 개수 세기
    label_counts = [np.count_nonzero(class_labels == label) for label in range(10)]

    indices_labels = []
    for label in range(10):
        indices = np.where(y_train == label)[0]
        indices_labels.append(indices)

    for i, classes in enumerate(class_labels):
        for label in classes:
            indices = indices_labels[label]
            num_images = int(6000 / label_counts[label])
            indices_ran = np.random.choice(indices, size=num_images, replace=True)
            indices_labels[label] = np.delete(indices, np.where(np.isin(indices, indices_ran)))

            dict_users[i] = np.concatenate((dict_users[i], indices_ran), axis=0)
    return dict_users , class_labels  # 딕셔너리 유저 반납

def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

def mnist_test_noniid(dataset,num_users,class_labels):
    # 각 클래스(또는 레이블)의 개수 세기
    label_counts = [np.count_nonzero(class_labels == label) for label in range(10)]
    test_label = dataset.test_labels

    indices_labels = []
    rhos = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    dict_rhos_groups = {i: np.array([]) for i in rhos}

    for label in range(10):
        indices = np.where(test_label == label)[0]
        indices_labels.append(indices)

    for rho in rhos: #각 rho 마다 local test set 만들기
        if rho == 0:
            dict_users = {i: np.array([]) for i in range(num_users)}
            for i, classes in enumerate(class_labels):
                for label in classes:  # 해당 label (class) 전체 데이터 세트가 들어 가게 된다.
                    indices = indices_labels[label]

                    dict_users[i] = np.concatenate((dict_users[i], indices), axis=0)

        else:
            dict_users = {i: np.array([]) for i in range(num_users)}

            for i, classes in enumerate(class_labels):
                total_label = [n_label for n_label in range(10)]
                rho_index = []

                for label in classes:  # 해당 label (class) 전체 데이터 세트가 들어 가게 된다.
                    indices = indices_labels[label]
                    dict_users[i] = np.concatenate((dict_users[i], indices), axis=0)
                    total_label = np.delete(total_label, np.where(np.isin(total_label, label)))

                for label in total_label:
                    indices = np.where(test_label == label)[0]
                    rho_index.append(indices)
                rho_index = np.array(np.concatenate(rho_index).ravel().tolist())

                indices_ran = np.random.choice(rho_index, size=int(2000 * rho), replace=True)
                dict_users[i] = np.concatenate((dict_users[i], indices_ran), axis=0)
        dict_rhos_groups[rho] = dict_users


    return dict_rhos_groups