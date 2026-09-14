import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import coo_matrix

from cgsr.layers import BipartiteGCNConv
from cgsr.losses import BPRLoss, EmbLoss, xavier_normal_initialization, xavier_uniform_initialization
from cgsr.recommender import SocialRecommender

class GatingLayer(nn.Module):
    def __init__(self, dim):
        super(GatingLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(self.dim, self.dim)
        self.activation = nn.Sigmoid()

    def forward(self, emb):
        embedding = self.linear(emb)
        embedding = self.activation(embedding)
        embedding = torch.mul(emb, embedding)
        return embedding

class AttLayer(nn.Module):
    def __init__(self, dim):
        super(AttLayer, self).__init__()
        self.dim = dim
        self.attention_mat = nn.Parameter(torch.randn([self.dim, self.dim]))
        self.attention = nn.Parameter(torch.randn([1, self.dim]))

    def forward(self, *embs):
        weights = []
        emb_list = []
        for embedding in embs:
            weights.append(torch.sum(torch.mul(self.attention, torch.matmul(embedding, self.attention_mat)), dim=1))
            emb_list.append(embedding)
        score = torch.nn.Softmax(dim=0)(torch.stack(weights, dim=0))
        embeddings = torch.stack(emb_list, dim=0)
        mixed_embeddings = torch.mul(embeddings, score.unsqueeze(dim=2).repeat(1, 1, self.dim)).sum(dim=0)
        return mixed_embeddings


class CGSR(SocialRecommender):

    input_type = "pairwise"

    def __init__(self, config, dataset):
        super(CGSR, self).__init__(config, dataset)

        self.R_user_edge_index, self.R_user_edge_weight, self.R_item_edge_index, self.R_item_edge_weight = self.get_bipartite_inter_mat(dataset)
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        if config['generator'] != 'GCN':
            raise ValueError('Final CGSR code keeps only generator=GCN.')
        if not config['cf_pos_flag'] or config['cf_neg_flag']:
            raise ValueError('Final CGSR code keeps cf_pos_flag=True and cf_neg_flag=False.')
        self.cf_pos_pos_weight = config['cf_pos_pos_weight']
        self.cf_pos_neg_weight = config['cf_pos_neg_weight']
        self.ib_beta = config['ib_beta']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.bipartite_gcn_conv = BipartiteGCNConv(dim=self.embedding_size)
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.ssl_reg = config['ssl_reg']
        self.reg_weight = config['reg_weight']

        H_s, H_j, H_p = self.get_motif_adj_matrix(dataset)
        self.H_s_edge_index, self.H_s_edge_weight = self.get_edge_index_weight(H_s)
        self.H_j_edge_index, self.H_j_edge_weight = self.get_edge_index_weight(H_j)
        self.H_p_edge_index, self.H_p_edge_weight = self.get_edge_index_weight(H_p)
        self.gating_c1 = GatingLayer(self.embedding_size)
        self.gating_c2 = GatingLayer(self.embedding_size)
        self.gating_c3 = GatingLayer(self.embedding_size)
        self.gating_simple = GatingLayer(self.embedding_size)
        self.ss_gating_c1 = GatingLayer(self.embedding_size)
        self.ss_gating_c2 = GatingLayer(self.embedding_size)
        self.ss_gating_c3 = GatingLayer(self.embedding_size)
        self.attention_layer = AttLayer(self.embedding_size)

        self.restore_user_e = None
        self.restore_item_e = None

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_bipartite_inter_mat(self, dataset):
        R_user_edge_index, R_user_edge_weight = dataset.get_bipartite_inter_mat(row='user', row_norm=False)
        R_item_edge_index, R_item_edge_weight = dataset.get_bipartite_inter_mat(row='item', row_norm=False)
        return R_user_edge_index.to(self.device), R_user_edge_weight.to(self.device), R_item_edge_index.to(self.device), R_item_edge_weight.to(self.device)

    def get_edge_index_weight(self, matrix):
        matrix = coo_matrix(matrix)
        edge_index = torch.stack([torch.LongTensor(matrix.row), torch.LongTensor(matrix.col)])
        edge_weight = torch.FloatTensor(matrix.data)
        return edge_index.to(self.device), edge_weight.to(self.device)

    def get_motif_adj_matrix(self, dataset):
        S = dataset.net_matrix()
        Y = dataset.inter_matrix()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9 + A9.T
        A10  = Y.dot(Y.T) - A8 - A9

        H_s = sum([A1, A2, A3, A4, A5, A6, A7])
        H_s = H_s.multiply(1.0 / (H_s.sum(axis=1) + 1e-7).reshape(-1, 1))
        H_j = sum([A8, A9])
        H_j = H_j.multiply(1.0 / (H_j.sum(axis=1) + 1e-7).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p > 1)
        H_p = H_p.multiply(1.0 / (H_p.sum(axis=1) + 1e-7).reshape(-1, 1))
        return H_s, H_j, H_p

    def forward(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        user_embeddings_c1 = self.gating_c1(user_embeddings)
        user_embeddings_c2 = self.gating_c2(user_embeddings)
        user_embeddings_c3 = self.gating_c3(user_embeddings)
        simple_user_embeddings = self.gating_simple(user_embeddings)

        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_c3 = [user_embeddings_c3]
        all_embeddings_simple = [simple_user_embeddings]
        all_embeddings_i = [item_embeddings]

        for layer_idx in range(self.n_layers):
            mixed_embedding = self.attention_layer(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3) + simple_user_embeddings / 2

            user_embeddings_c1 = self.bipartite_gcn_conv((user_embeddings_c1, user_embeddings_c1), self.H_s_edge_index.flip([0]), self.H_s_edge_weight, size=(self.n_users, self.n_users))
            norm_embeddings = F.normalize(user_embeddings_c1, p=2, dim=1)
            all_embeddings_c1 += [norm_embeddings]

            user_embeddings_c2 = self.bipartite_gcn_conv((user_embeddings_c2, user_embeddings_c2), self.H_j_edge_index.flip([0]), self.H_j_edge_weight, size=(self.n_users, self.n_users))
            norm_embeddings = F.normalize(user_embeddings_c2, p=2, dim=1)
            all_embeddings_c2 += [norm_embeddings]

            user_embeddings_c3 = self.bipartite_gcn_conv((user_embeddings_c3, user_embeddings_c3), self.H_p_edge_index.flip([0]), self.H_p_edge_weight, size=(self.n_users, self.n_users))
            norm_embeddings = F.normalize(user_embeddings_c3, p=2, dim=1)
            all_embeddings_c3 += [norm_embeddings]

            new_item_embeddings = self.bipartite_gcn_conv((mixed_embedding, item_embeddings), self.R_item_edge_index.flip([0]), self.R_item_edge_weight, size=(self.n_users, self.n_items))
            norm_embeddings = F.normalize(new_item_embeddings, p=2, dim=1)
            all_embeddings_i += [norm_embeddings]
            simple_user_embeddings = self.bipartite_gcn_conv((item_embeddings, simple_user_embeddings), self.R_user_edge_index.flip([0]), self.R_user_edge_weight, size=(self.n_items, self.n_users))
            norm_embeddings = F.normalize(simple_user_embeddings, p=2, dim=1)
            all_embeddings_simple += [norm_embeddings]
            item_embeddings = new_item_embeddings

        user_embeddings_c1 = torch.stack(all_embeddings_c1, dim=0).sum(dim=0)
        user_embeddings_c2 = torch.stack(all_embeddings_c2, dim=0).sum(dim=0)
        user_embeddings_c3 = torch.stack(all_embeddings_c3, dim=0).sum(dim=0)
        simple_user_embeddings = torch.stack(all_embeddings_simple, dim=0).sum(dim=0)
        item_all_embeddings = torch.stack(all_embeddings_i, dim=0).sum(dim=0)

        user_all_embeddings = self.attention_layer(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)
        user_all_embeddings += simple_user_embeddings / 2

        return user_all_embeddings, item_all_embeddings

    def hierarchical_self_supervision(self, user_embeddings, edge_index, edge_weight):
        def row_shuffle(embedding):
            shuffled_embeddings = embedding[torch.randperm(embedding.size(0))]
            return shuffled_embeddings
        def row_column_shuffle(embedding):
            shuffled_embeddings = embedding[:, torch.randperm(embedding.size(1))]
            shuffled_embeddings = shuffled_embeddings[torch.randperm(embedding.size(0))]
            return shuffled_embeddings
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=1)

        edge_embeddings = self.bipartite_gcn_conv((user_embeddings, user_embeddings), edge_index.flip([0]), edge_weight, size=(self.n_users, self.n_users))
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)) - torch.log(torch.sigmoid(neg1 - neg2)))
        graph = torch.mean(edge_embeddings, dim=0, keepdim=True)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos - neg1)))
        return global_loss + local_loss

    def calculate_loss(self, interaction, user_all_embeddings=None, item_all_embeddings=None):
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        u_embeddings = user_all_embeddings[users]
        posi_embeddings = item_all_embeddings[pos_items]
        negi_embeddings = item_all_embeddings[neg_items]

        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)
        ss_loss = self.hierarchical_self_supervision(self.ss_gating_c1(user_all_embeddings), self.H_s_edge_index, self.H_s_edge_weight)
        ss_loss += self.hierarchical_self_supervision(self.ss_gating_c2(user_all_embeddings), self.H_j_edge_index, self.H_j_edge_weight)
        ss_loss += self.hierarchical_self_supervision(self.ss_gating_c3(user_all_embeddings), self.H_p_edge_index, self.H_p_edge_weight)
        u_ego_embeddings = self.user_embedding(users)
        pos_ego_embeddings = self.item_embedding(pos_items)
        neg_ego_embeddings = self.item_embedding(neg_items)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        main_loss = bpr_loss + self.ssl_reg * ss_loss + self.reg_weight * reg_loss

        cf_u_embeddings = self.get_cf_u_embeddings_GCN(
            interaction['cf_pos_kg_neighbors'],
            user_all_embeddings,
            item_all_embeddings,
        )
        cf_posu_embeddings = cf_u_embeddings[users]
        cf_pos_scores_pos = torch.mul(cf_posu_embeddings, posi_embeddings).sum(dim=1)
        cf_pos_scores_neg = torch.mul(cf_posu_embeddings, negi_embeddings).sum(dim=1)
        cf_pos_pos_loss = self.bpr_loss(cf_pos_scores_pos, neg_scores)
        cf_pos_neg_loss = self.bpr_loss(pos_scores, cf_pos_scores_neg)
        cf_loss = self.cf_pos_pos_weight * cf_pos_pos_loss + self.cf_pos_neg_weight * cf_pos_neg_loss

        return main_loss, cf_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]

        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

    def get_cf_u_embeddings_GCN(self, kg_neighbors, user_embedding, item_embedding):
        kg_neighbors = kg_neighbors.long()
        user_idx, item_idx = self.R_user_edge_index
        user_num = user_embedding.shape[0]
        emb_dim = user_embedding.shape[1]
        user_emb = torch.zeros(user_num, emb_dim, device=item_embedding.device)
        user_emb.index_add_(0, user_idx, item_embedding[item_idx])
        deg = torch.bincount(user_idx, minlength=user_num).clamp(min=1).unsqueeze(1)
        user_interaction_emb = user_emb / deg

        friend_emb = user_embedding[kg_neighbors]
        user_social_emb = torch.mean(friend_emb, dim=1)

        user_embeddings = (user_interaction_emb + user_social_emb) / 2

        return user_embeddings

    def get_reward1(self, scores1, scores2, embeddings):
        gamma = 1e-10
        part1 = torch.log(gamma + torch.sigmoid(scores1 - scores2))
        part2 = - torch.norm(embeddings, dim=1, p=2)
        return part1 + self.ib_beta * part2

    def generate_pos_reward(self, interaction, kg_neighbors1, user_all_embeddings, item_all_embeddings):
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        u_embeddings = user_all_embeddings[users]
        posi_embeddings = item_all_embeddings[pos_items]
        negi_embeddings = item_all_embeddings[neg_items]
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)

        cf_u_embeddings1 = self.get_cf_u_embeddings_GCN(kg_neighbors1, user_all_embeddings, item_all_embeddings)
        cf_posu_embeddings1 = cf_u_embeddings1[users]
        cf_pos_pos_scores1 = torch.mul(cf_posu_embeddings1, posi_embeddings).sum(dim=1)
        cf_pos_neg_scores1 = torch.mul(cf_posu_embeddings1, negi_embeddings).sum(dim=1)
        reward1 = self.get_reward1(cf_pos_pos_scores1, neg_scores, cf_posu_embeddings1)
        reward1 += self.get_reward1(pos_scores, cf_pos_neg_scores1, cf_posu_embeddings1)

        return reward1


class CFGenerator(nn.Module):

    def __init__(self, config, dataset):
        super(CFGenerator, self).__init__()

        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.device = config['device']

        self.all_user_id = dataset.dataset.inter_feat.interaction[config['USER_ID_FIELD']].unsqueeze(1).to(self.device)
        self.all_item_id = dataset.dataset.inter_feat.interaction[config['ITEM_ID_FIELD']].unsqueeze(1).to(self.device)
        self.inter_edge = torch.cat((self.all_user_id, self.all_item_id), dim=1).to(self.device)
        self.embedding_size = config['embedding_size']
        self.replace_num = config['replace_num']
        self.cf_generate_type = config['cf_generate_type']

        self.ui_linear = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.e_step1_linear = nn.Linear(self.embedding_size, self.embedding_size)
        self.step1_softmax = nn.Softmax(dim=1)

        self.apply(xavier_normal_initialization)

    def generate(self, kg_neighbors, user_all_embeddings=None, item_all_embeddings=None, user_embeddings=None):
        raise NotImplementedError()


class CFPosGenerator(CFGenerator):

    def __init__(self, config, dataset):
        super(CFPosGenerator, self).__init__(config, dataset)

    def generate(self, kg_neighbors, user_all_embeddings=None, item_all_embeddings=None, user_embeddings=None):

        device = self.device
        eps = 1e-12

        kg_neighbors = kg_neighbors.long().to(device)
        num_users, n_cans = kg_neighbors.size()

        u_idx = self.all_user_id.squeeze().long().to(device)
        i_idx = self.all_item_id.squeeze().long().to(device)
        E = u_idx.size(0)

        neighbors_e_u = F.normalize(self.e_step1_linear(user_all_embeddings[kg_neighbors]), p=2, dim=2)

        neighbors_e_edges = neighbors_e_u.index_select(0, u_idx)

        ui_input = torch.cat([user_embeddings[u_idx], item_all_embeddings[i_idx]], dim=1)
        ui_e_edge = F.normalize(self.ui_linear(ui_input), p=2, dim=1).unsqueeze(1)

        scores = torch.bmm(ui_e_edge, neighbors_e_edges.transpose(1, 2)).squeeze(1)
        logits = self.step1_softmax(scores)


        neg_inf = torch.tensor(-float("inf"), device=device, dtype=scores.dtype)
        max_scores_uc = torch.full((num_users, n_cans), neg_inf, device=device, dtype=scores.dtype)
        argmax_row_uc = torch.zeros((num_users, n_cans), device=device, dtype=torch.long)

        sorted_u, perm = torch.sort(u_idx)
        scores_sorted = scores.index_select(0, perm)

        if E > 0:
            diff = sorted_u[1:] != sorted_u[:-1]
            boundary = torch.nonzero(diff, as_tuple=False).squeeze(1) + 1

            starts = torch.cat([torch.zeros(1, device=device, dtype=torch.long), boundary])
            ends = torch.cat([boundary, torch.tensor([E], device=device, dtype=torch.long)])

            uniq_users = sorted_u.index_select(0, starts)
            lengths = (ends - starts).long()
            max_len = int(lengths.max().item())

            pos = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0).expand(uniq_users.numel(), max_len)
            mask = pos < lengths.unsqueeze(1)

            global_row = starts.unsqueeze(1) + pos
            global_row = global_row.clamp(max=E - 1)

            scores_pad = scores_sorted.index_select(0, global_row.reshape(-1)).reshape(uniq_users.numel(), max_len, n_cans)
            scores_pad = torch.where(mask.unsqueeze(2), scores_pad, neg_inf)

            vals, local_pos = torch.max(scores_pad, dim=1)

            chosen_sorted_row = starts.unsqueeze(1) + local_pos
            rows = perm.index_select(0, chosen_sorted_row.reshape(-1)) \
                .reshape(uniq_users.numel(), n_cans)

            valid_mask = (kg_neighbors.index_select(0, uniq_users) != 0)
            vals = torch.where(valid_mask, vals, neg_inf)
            rows = torch.where(valid_mask, rows, torch.zeros_like(rows))

            max_scores_uc.index_copy_(0, uniq_users, vals)
            argmax_row_uc.index_copy_(0, uniq_users, rows)


        flat_global = max_scores_uc.reshape(-1)
        finite_mask = torch.isfinite(flat_global)
        valid_cnt = int(finite_mask.sum().item())

        cf_kg_neighbors1 = kg_neighbors.clone()
        base_dtype = logits.dtype
        log_eps = torch.log(torch.tensor(eps, device=device, dtype=base_dtype))
        action_prob1 = torch.full((num_users,), log_eps, device=device, dtype=base_dtype)

        hit_cnt = torch.zeros((num_users,), device=device, dtype=torch.long)

        if valid_cnt > 0:
            k = min(self.replace_num, valid_cnt)
            _, top_idx = torch.topk(flat_global, k=k, largest=True, sorted=False)
            sel_users = (top_idx // n_cans).long()
            sel_cols = (top_idx % n_cans).long()

            cf_kg_neighbors1[sel_users, sel_cols] = 0

            sel_rows = argmax_row_uc[sel_users, sel_cols].clamp(min=0, max=max(E - 1, 0))
            prob = logits[sel_rows, sel_cols]
            sel_log_probs = torch.log(prob.clamp_min(eps))

            sum_logprob = torch.zeros((num_users,), device=device, dtype=base_dtype).scatter_add(0, sel_users, sel_log_probs)
            hit_cnt = hit_cnt.scatter_add(0, sel_users, torch.ones_like(sel_users, dtype=torch.long))
            action_prob1 = torch.where(hit_cnt > 0, sum_logprob, action_prob1)

        return cf_kg_neighbors1, action_prob1, hit_cnt
