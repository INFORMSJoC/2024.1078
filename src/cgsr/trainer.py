import copy
from time import time

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.optim as optim
from tqdm import tqdm


class CGSRTrainer:
    def __init__(self, config, dataset, rec_model, raw_kg_neighbors, cf_pos_generator=None):
        self.config = config
        self.model = rec_model
        self.device = config.device
        self.USER_ID = config.USER_ID_FIELD
        self.ITEM_ID = config.ITEM_ID_FIELD
        self.NEG_ITEM_ID = config.NEG_PREFIX + self.ITEM_ID

        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.epochs = config.epochs
        self.eval_step = min(config.eval_step, config.epochs) if config.epochs > 0 else config.eval_step
        self.stopping_step = config.stopping_step
        self.valid_metric = config.valid_metric.lower()
        self.enable_amp = config.enable_amp
        self.enable_scaler = torch.cuda.is_available() and config.enable_scaler

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.best_valid_score = -np.inf
        self.best_valid_result = None
        self.best_model_state = None
        self.cur_step = 0

        self.glr = config.glr
        self.gamma = config.gamma
        self.cf_pos_flag = config.cf_pos_flag
        self.cf_neg_flag = config.cf_neg_flag
        self.max_neighbor_size = config.max_neighbor_size
        self.replace_step = config.replace_step
        self.train_recommender = config.train_recommender
        self.train_generator = config.train_generator
        self.generator = config.generator
        if self.generator != "GCN":
            raise ValueError("Standalone CGSR keeps only generator=GCN.")
        if not self.cf_pos_flag or self.cf_neg_flag:
            raise ValueError("Standalone CGSR keeps cf_pos_flag=True and cf_neg_flag=False.")

        self.n_users = dataset.dataset.user_num
        self.n_items = dataset.dataset.item_num
        self.n_nets = dataset.dataset.net_num
        self.kg_neighbors = torch.from_numpy(raw_kg_neighbors).to(self.device)

        self.cf_pos_generator = cf_pos_generator
        self.cf_pos_optimizer = optim.Adam(self.cf_pos_generator.parameters(), lr=self.glr)

    def generate_cf_net(self, interaction, user_all_embeddings, item_all_embeddings):
        kg_neighbors = self.kg_neighbors.clone()
        for _ in range(self.replace_step):
            user_embeddings = self.model.get_cf_u_embeddings_GCN(kg_neighbors, user_all_embeddings, item_all_embeddings)
            kg_neighbors, _, _ = self.cf_pos_generator.generate(
                kg_neighbors, user_all_embeddings, item_all_embeddings, user_embeddings
            )
            user_all_embeddings = user_embeddings
        return kg_neighbors

    def calculate_generator_loss(self, interaction, user_all_embeddings, item_all_embeddings):
        users = interaction[self.USER_ID]
        kg_neighbors = self.kg_neighbors.clone()
        batch_g_loss = None
        for i in range(self.replace_step):
            with torch.no_grad():
                user_embeddings = self.model.get_cf_u_embeddings_GCN(
                    kg_neighbors, user_all_embeddings, item_all_embeddings
                )
            kg_neighbors, action_prob_all, hit_cnt_all = self.cf_pos_generator.generate(
                kg_neighbors, user_all_embeddings, item_all_embeddings, user_embeddings
            )
            with torch.no_grad():
                step_reward1 = self.model.generate_pos_reward(
                    interaction, kg_neighbors, user_all_embeddings, item_all_embeddings
                )
            user_all_embeddings = user_embeddings
            action_prob1 = action_prob_all[users]
            hit_cnt = hit_cnt_all[users]
            hit_mask = hit_cnt > 0
            if hit_mask.any():
                step_loss = -(step_reward1[hit_mask] * action_prob1[hit_mask]).mean()
            else:
                step_loss = torch.tensor(0.0, device=self.device)
            step_loss = self.gamma ** i * step_loss
            batch_g_loss = step_loss if batch_g_loss is None else batch_g_loss + step_loss
        return batch_g_loss / self.replace_step

    def _train_epoch(self, train_data, epoch_idx, show_progress=True):
        total_loss = []
        total_base_loss, total_pos_g_loss = None, None

        if self.train_recommender:
            self.model.train()
            iter_data = tqdm(train_data, total=len(train_data), ncols=100, desc=f"Train {epoch_idx:>5}") if show_progress else train_data
            scaler = amp.GradScaler(enabled=self.enable_scaler)
            for interaction in iter_data:
                interaction = interaction.to(self.device)
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                    user_all_embeddings, item_all_embeddings = self.model.forward()
                with torch.no_grad():
                    kg_neighbors = self.generate_cf_net(interaction, user_all_embeddings, item_all_embeddings)
                    interaction.interaction["cf_pos_kg_neighbors"] = kg_neighbors
                with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                    losses = self.model.calculate_loss(interaction, user_all_embeddings, item_all_embeddings)
                loss = sum(losses)
                self._check_nan(loss)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_base_loss = loss_tuple if total_base_loss is None else tuple(map(sum, zip(total_base_loss, loss_tuple)))

        if self.train_generator:
            with torch.no_grad():
                user_all_embeddings, item_all_embeddings = self.model.forward()

            self.cf_pos_generator.train()
            iter_data = tqdm(train_data, total=len(train_data), ncols=100, desc=f"CFPos Train {epoch_idx:>5}") if show_progress else train_data
            scaler2 = amp.GradScaler(enabled=self.enable_scaler)
            for interaction in iter_data:
                interaction = interaction.to(self.device)
                self.cf_pos_optimizer.zero_grad()
                batch_pos_g_loss = self.calculate_generator_loss(interaction, user_all_embeddings, item_all_embeddings)
                self._check_nan(batch_pos_g_loss)
                scaler2.scale(batch_pos_g_loss).backward()
                scaler2.step(self.cf_pos_optimizer)
                scaler2.update()
                total_pos_g_loss = batch_pos_g_loss if total_pos_g_loss is None else total_pos_g_loss + batch_pos_g_loss

        if self.train_recommender:
            total_loss.extend(total_base_loss)
        if self.train_generator:
            total_loss.append(float(total_pos_g_loss.detach().cpu()))

        kg_neighbors_np, _ = train_data.dataset.user_neighbors(self.max_neighbor_size)
        self.kg_neighbors = torch.from_numpy(kg_neighbors_np).to(self.device)
        return tuple(total_loss)

    def fit(self, train_data, valid_data=None, show_progress=True):
        for epoch_idx in range(self.epochs):
            start = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            print(f"epoch {epoch_idx} training [time: {time() - start:.2f}s, losses: {train_loss}]")

            if valid_data is None or self.eval_step <= 0 or (epoch_idx + 1) % self.eval_step != 0:
                continue

            valid_start = time()
            valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
            valid_score = valid_result[self.valid_metric]
            print(f"epoch {epoch_idx} evaluating [time: {time() - valid_start:.2f}s, valid_score: {valid_score:.6f}]")
            print("valid result:", _dict_to_str(valid_result))

            if valid_score >= self.best_valid_score:
                self.best_valid_score = valid_score
                self.best_valid_result = valid_result
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.cur_step = 0
            else:
                self.cur_step += 1

            if self.cur_step > self.stopping_step:
                print(f"Finished training, best eval result before epoch {epoch_idx}")
                break

        return self.best_valid_score, self.best_valid_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, show_progress=False):
        if load_best_model and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.load_other_parameter(None)

        self.model.eval()
        self.model.restore_user_e, self.model.restore_item_e = None, None

        result_accumulator = RankingAccumulator(self.config.topk)
        iter_data = tqdm(eval_data, total=len(eval_data), ncols=100, desc="Evaluate") if show_progress else eval_data
        for interaction, history_index, positive_u, positive_i in iter_data:
            interaction = interaction.to(self.device)
            scores = self.model.full_sort_predict(interaction).view(-1, eval_data.dataset.item_num)
            scores[:, 0] = -np.inf
            if history_index is not None:
                history_u, history_i = history_index
                scores[(history_u.to(self.device), history_i.to(self.device))] = -np.inf
            result_accumulator.collect(scores, positive_u.to(self.device), positive_i.to(self.device))
        return result_accumulator.result(self.config.metric_decimal_place)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

class RankingAccumulator:
    def __init__(self, topk):
        self.topk = tuple(sorted(topk))
        self.max_k = max(self.topk)
        self.rows = []

    def collect(self, scores, positive_u, positive_i):
        if len(positive_u) == 0:
            return
        _, top_items = torch.topk(scores, k=self.max_k, dim=1)
        positives = {}
        for row, item in zip(positive_u.tolist(), positive_i.tolist()):
            positives.setdefault(int(row), set()).add(int(item))

        for row, pos_items in positives.items():
            ranked = top_items[row].tolist()
            self.rows.append((ranked, pos_items))

    def result(self, decimals):
        metric_sum = {f"{metric}@{k}": 0.0 for metric in ("recall", "mrr", "ndcg", "hit", "precision") for k in self.topk}
        user_count = len(self.rows)
        if user_count == 0:
            return {key: 0.0 for key in metric_sum}

        for ranked, pos_items in self.rows:
            pos_len = len(pos_items)
            for k in self.topk:
                top = ranked[:k]
                hits = [1 if item in pos_items else 0 for item in top]
                hit_count = sum(hits)
                metric_sum[f"recall@{k}"] += hit_count / pos_len
                metric_sum[f"precision@{k}"] += hit_count / k
                metric_sum[f"hit@{k}"] += 1.0 if hit_count > 0 else 0.0
                metric_sum[f"mrr@{k}"] += _mrr(hits)
                metric_sum[f"ndcg@{k}"] += _ndcg(hits, min(pos_len, k))

        return {key: round(value / user_count, decimals) for key, value in metric_sum.items()}


def _mrr(hits):
    for idx, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / idx
    return 0.0


def _ndcg(hits, ideal_len):
    dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))
    if ideal_len == 0:
        return 0.0
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_len))
    return dcg / idcg


def _dict_to_str(result):
    return "    ".join(f"{metric} : {value}" for metric, value in result.items())
