import numpy as np
from torchvision import datasets, transforms

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
    num_shards, num_imgs = 200,300
    idx_shard = [i for i in range(num_shards)] # list에 shrads의 개수만큼 index 생성
    dict_users = {i : np.array([]) for i in range(num_users)} # 유저의 수 만큼 딕셔너리 생성
    idxs = np.arange(num_shards*num_imgs) # shard이 개수와 이미지의 개수를 곱한 수 만큼의 list 생성 (200*300=> [0,1,,,,,59999]
    labels = dataset.train_labels.numpy() # dataset.train_labels를 numpy 배열로 반환한다.

    #labels 정렬(sort)
    idxs_labels = np.vstack((idxs,labels)) #idx와 labels의 결합 ([[idxs],[labels]])
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()] #idxs_labels에서 index는 그대로 두고 labels만 오름차순으로 정렬
    idxs = idxs_labels[0, :] #idxs는 따로 정렬된 새로 운 idx로 저장

    #각 client당 2개의 shards로 분배
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace = False))
        idx_shard = list(set(idx_shard)-rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i],idxs[rand*num_imgs:(rand+1)*num_imgs]),axis = 0)
            # numpy 합침 // {각 유저 딕셔너리 index i : 뽑아진 각 shard인덱스에 맞게 imgs 개수를 곱해서 해당 index를 함께 저장}
    return dict_users  # 딕셔너리 유저 반납

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