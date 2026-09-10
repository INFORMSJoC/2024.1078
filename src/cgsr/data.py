import collections
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix


class Interaction:
    def __init__(self, data):
        self.interaction = {
            key: value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.long)
            for key, value in data.items()
        }

    def __getitem__(self, key):
        return self.interaction[key]

    def __setitem__(self, key, value):
        self.interaction[key] = value

    def __contains__(self, key):
        return key in self.interaction

    def __len__(self):
        if not self.interaction:
            return 0
        return len(next(iter(self.interaction.values())))

    @property
    def length(self):
        return len(self)

    def subset(self, index):
        return Interaction({key: value[index] for key, value in self.interaction.items()})

    def to(self, device):
        for key, value in self.interaction.items():
            self.interaction[key] = value.to(device)
        return self


class CGSRDataset:
    def __init__(self, config, inter_feat, net_feat, user_num, item_num, user_token2id, item_token2id):
        self.config = config
        self.uid_field = config.USER_ID_FIELD
        self.iid_field = config.ITEM_ID_FIELD
        self.net_src_field = config.NET_SOURCE_ID_FIELD
        self.net_tgt_field = config.NET_TARGET_ID_FIELD

        self.inter_feat = inter_feat
        self.net_feat = net_feat
        self.user_num = user_num
        self.item_num = item_num
        self.user_token2id = user_token2id
        self.item_token2id = item_token2id

    @classmethod
    def load(cls, config):
        dataset_dir = Path(config.dataset_dir)
        inter_path = dataset_dir / f"{config.dataset}.inter"
        net_path = dataset_dir / f"{config.dataset}.net"
        inter_df = _read_token_table(inter_path)
        net_df = _read_token_table(net_path)

        inter_users = set(inter_df[config.USER_ID_FIELD].astype(str))
        net_df = net_df[
            net_df[config.NET_SOURCE_ID_FIELD].astype(str).isin(inter_users)
            & net_df[config.NET_TARGET_ID_FIELD].astype(str).isin(inter_users)
        ].copy()

        user_token2id = _build_mapping(
            [
                inter_df[config.USER_ID_FIELD],
                net_df[config.NET_SOURCE_ID_FIELD],
                net_df[config.NET_TARGET_ID_FIELD],
            ]
        )
        item_token2id = _build_mapping([inter_df[config.ITEM_ID_FIELD]])

        inter_feat = Interaction(
            {
                config.USER_ID_FIELD: _map_series(inter_df[config.USER_ID_FIELD], user_token2id),
                config.ITEM_ID_FIELD: _map_series(inter_df[config.ITEM_ID_FIELD], item_token2id),
            }
        )
        net_feat = Interaction(
            {
                config.NET_SOURCE_ID_FIELD: _map_series(net_df[config.NET_SOURCE_ID_FIELD], user_token2id),
                config.NET_TARGET_ID_FIELD: _map_series(net_df[config.NET_TARGET_ID_FIELD], user_token2id),
            }
        )
        return cls(
            config=config,
            inter_feat=inter_feat,
            net_feat=net_feat,
            user_num=len(user_token2id),
            item_num=len(item_token2id),
            user_token2id=user_token2id,
            item_token2id=item_token2id,
        )

    def copy_with_interactions(self, inter_feat):
        return CGSRDataset(
            config=self.config,
            inter_feat=inter_feat,
            net_feat=self.net_feat,
            user_num=self.user_num,
            item_num=self.item_num,
            user_token2id=self.user_token2id,
            item_token2id=self.item_token2id,
        )

    def __len__(self):
        return len(self.inter_feat)

    @property
    def inter_num(self):
        return len(self.inter_feat)

    @property
    def net_num(self):
        return len(self.net_feat)

    def num(self, field):
        if field == self.uid_field:
            return self.user_num
        if field == self.iid_field:
            return self.item_num
        raise KeyError(field)

    def __getitem__(self, index):
        return self.inter_feat.subset(index)

    def split(self, ratios):
        ratios = np.asarray(ratios, dtype=np.float64)
        ratios = ratios / ratios.sum()

        total = len(self)
        shuffled = torch.randperm(total)
        shuffled_inter = self.inter_feat.subset(shuffled)

        grouped = collections.OrderedDict()
        for idx, user in enumerate(shuffled_inter[self.uid_field].numpy()):
            grouped.setdefault(int(user), []).append(idx)

        split_indexes = [[] for _ in ratios]
        for grouped_index in grouped.values():
            split_ids = _split_ids(len(grouped_index), ratios)
            for target, start, end in zip(split_indexes, [0] + split_ids, split_ids + [len(grouped_index)]):
                target.extend(grouped_index[start:end])

        datasets = []
        for index in split_indexes:
            tensor_index = torch.as_tensor(index, dtype=torch.long)
            datasets.append(self.copy_with_interactions(shuffled_inter.subset(tensor_index)))
        return datasets

    def get_bipartite_inter_mat(self, row="user", row_norm=True):
        if row == "user":
            row_field, col_field = self.uid_field, self.iid_field
        else:
            row_field, col_field = self.iid_field, self.uid_field

        row_tensor = self.inter_feat[row_field].long()
        col_tensor = self.inter_feat[col_field].long()
        edge_index = torch.stack([row_tensor, col_tensor])

        if row_norm:
            deg = torch.bincount(row_tensor, minlength=self.num(row_field)).float()
            norm_deg = 1.0 / torch.where(deg == 0, torch.ones_like(deg), deg)
            edge_weight = norm_deg[row_tensor]
        else:
            row_deg = torch.bincount(row_tensor, minlength=self.num(row_field)).float()
            col_deg = torch.bincount(col_tensor, minlength=self.num(col_field)).float()
            row_norm_deg = 1.0 / torch.sqrt(torch.where(row_deg == 0, torch.ones_like(row_deg), row_deg))
            col_norm_deg = 1.0 / torch.sqrt(torch.where(col_deg == 0, torch.ones_like(col_deg), col_deg))
            edge_weight = row_norm_deg[row_tensor] * col_norm_deg[col_tensor]
        return edge_index, edge_weight

    def inter_matrix(self, form="coo"):
        matrix = coo_matrix(
            (
                np.ones(len(self.inter_feat)),
                (
                    self.inter_feat[self.uid_field].numpy(),
                    self.inter_feat[self.iid_field].numpy(),
                ),
            ),
            shape=(self.user_num, self.item_num),
        )
        return matrix.tocsr() if form == "csr" else matrix

    def net_matrix(self, form="coo"):
        matrix = coo_matrix(
            (
                np.ones(len(self.net_feat)),
                (
                    self.net_feat[self.net_src_field].numpy(),
                    self.net_feat[self.net_tgt_field].numpy(),
                ),
            ),
            shape=(self.user_num, self.user_num),
        )
        return matrix.tocsr() if form == "csr" else matrix

    def __str__(self):
        return "\n".join(
            [
                self.config.dataset,
                f"The number of users: {self.user_num}",
                f"The number of items: {self.item_num}",
                f"The number of inters: {self.inter_num}",
                f"The number of social network relations: {self.net_num}",
            ]
        )


class TrainDataLoader:
    def __init__(self, config, dataset, used_items, shuffle=True):
        self.config = config
        self.dataset = dataset
        self.used_items = used_items
        self.shuffle = shuffle
        self.batch_size = config.train_batch_size
        self.uid_field = config.USER_ID_FIELD
        self.iid_field = config.ITEM_ID_FIELD
        self.neg_iid_field = config.NEG_PREFIX + self.iid_field
        self.generator = torch.Generator()
        self.generator.manual_seed(config.seed)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            torch.empty((), dtype=torch.int64).random_(generator=self.generator)
            indices = torch.randperm(len(self.dataset), generator=self.generator).numpy()
        else:
            indices = np.arange(len(self.dataset))
        for start in range(0, len(indices), self.batch_size):
            batch_index = torch.as_tensor(indices[start:start + self.batch_size], dtype=torch.long)
            batch = self.dataset[batch_index]
            users = batch[self.uid_field].numpy()
            batch[self.neg_iid_field] = _sample_negative_items(users, self.dataset.item_num, self.used_items)
            yield batch
        if self.shuffle:
            torch.randperm(len(self.dataset), generator=self.generator)


class FullSortEvalDataLoader:
    def __init__(self, config, dataset, history_used_items):
        self.config = config
        self.dataset = dataset
        self.history_used_items = history_used_items
        self.uid_field = config.USER_ID_FIELD
        self.iid_field = config.ITEM_ID_FIELD

        positive = collections.defaultdict(set)
        for user, item in zip(dataset.inter_feat[self.uid_field].tolist(), dataset.inter_feat[self.iid_field].tolist()):
            positive[int(user)].add(int(item))
        self.positive_items = positive
        self.uid_list = sorted(positive.keys())
        self.batch_user_num = max(config.eval_batch_size // dataset.item_num, 1)

    def __len__(self):
        return math.ceil(len(self.uid_list) / self.batch_user_num)

    def __iter__(self):
        for start in range(0, len(self.uid_list), self.batch_user_num):
            uid_batch = self.uid_list[start:start + self.batch_user_num]
            interaction = Interaction({self.uid_field: torch.as_tensor(uid_batch, dtype=torch.long)})

            history_u, history_i, positive_u, positive_i = [], [], [], []
            for row, user in enumerate(uid_batch):
                for item in self.history_used_items[user] - self.positive_items[user]:
                    history_u.append(row)
                    history_i.append(item)
                for item in self.positive_items[user]:
                    positive_u.append(row)
                    positive_i.append(item)

            history_index = None
            if history_u:
                history_index = (
                    torch.as_tensor(history_u, dtype=torch.long),
                    torch.as_tensor(history_i, dtype=torch.long),
                )
            yield (
                interaction,
                history_index,
                torch.as_tensor(positive_u, dtype=torch.long),
                torch.as_tensor(positive_i, dtype=torch.long),
            )


def prepare_data(config):
    dataset = CGSRDataset.load(config)
    train_dataset, valid_dataset, test_dataset = dataset.split(config.split_ratios)

    train_used = _used_items(train_dataset, config, base=None)
    valid_used = _used_items(valid_dataset, config, base=train_used)
    test_used = _used_items(test_dataset, config, base=valid_used)

    train_data = TrainDataLoader(config, train_dataset, train_used, shuffle=config.shuffle)
    valid_data = FullSortEvalDataLoader(config, valid_dataset, valid_used)
    test_data = FullSortEvalDataLoader(config, test_dataset, test_used)
    return dataset, train_data, valid_data, test_data


def _read_token_table(path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [col.split(":")[0] for col in df.columns]
    return df


def _map_series(series, mapping):
    return torch.as_tensor([mapping[str(token)] for token in series], dtype=torch.long)


def _split_ids(total, ratios):
    cnt = [int(ratio * total) for ratio in ratios]
    cnt[0] = total - sum(cnt[1:])
    for i in range(1, len(ratios)):
        if cnt[0] <= 1:
            break
        if 0 < ratios[-i] * total < 1:
            cnt[-i] += 1
            cnt[0] -= 1
    return list(np.cumsum(cnt)[:-1])


def _used_items(dataset, config, base=None):
    if base is None:
        used = [set() for _ in range(dataset.user_num)]
    else:
        used = [set(items) for items in base]
    for user, item in zip(dataset.inter_feat[config.USER_ID_FIELD].tolist(), dataset.inter_feat[config.ITEM_ID_FIELD].tolist()):
        used[int(user)].add(int(item))
    return used


def _sample_negative_items(users, item_num, used_items):
    users = np.asarray(users)
    value_ids = np.zeros(len(users), dtype=np.int64)
    check_list = np.arange(len(users))
    while len(check_list) > 0:
        value_ids[check_list] = np.random.randint(1, item_num, len(check_list))
        check_list = np.array(
            [
                idx
                for idx, user, item in zip(check_list, users[check_list], value_ids[check_list])
                if int(item) in used_items[int(user)]
            ]
        )
    return torch.as_tensor(value_ids, dtype=torch.long)
