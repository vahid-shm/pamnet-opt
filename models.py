# -*- coding: utf-8 -*-
"""
models.py

تعریف مدل‌های PAMNet و PAMNet_s برای آزمایش‌های QM9.

این نسخه برای مقایسه‌ی چهار حالت معماری آماده شده است:
1. original: بدون weight sharing و بدون basis reduction
2. ws: استفاده از weight sharing در لایه‌های local/global
3. br: کاهش تعداد basis functions برای کم کردن محاسبات و پارامترهای وابسته
4. wsbr: ترکیب weight sharing و basis reduction

تفاوت اصلی با نسخه‌ی اولیه‌ی hard-code شده این است که اینجا تغییرات معماری از طریق Config کنترل می‌شوند،
پس برای اجرای حالت‌های مختلف لازم نیست داخل کد دستکاری دستی انجام شود.
"""

# -----------------------------
# کتابخانه‌های اصلی
# -----------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# SparseTensor برای ساخت adjacency sparse و استخراج indexهای triplet/pair استفاده می‌شود.
from torch_sparse import SparseTensor

# global_add_pool خروجی node-level را به graph-level تبدیل می‌کند.
# radius برای ساخت graph سراسری براساس فاصله‌ی سه‌بعدی اتم‌ها استفاده می‌شود.
from torch_geometric.nn import global_add_pool, radius
from torch_geometric.utils import remove_self_loops

# لایه‌های اصلی message passing و basis embedding از فایل layers.py پروژه می‌آیند.
from layers import (
    Global_MessagePassing,
    Local_MessagePassing,
    Local_MessagePassing_s,
    BesselBasisLayer,
    SphericalBasisLayer,
    MLP,
)


class Config(object):
    """
    کلاس نگه‌دارنده‌ی تنظیمات مدل.

    هدف این کلاس این است که همه‌ی hyperparameterهای معماری در یک object جمع شوند و به PAMNet/PAMNet_s داده شوند.

    پارامترهای مهم:
    - dim: اندازه‌ی hidden representation هر اتم
    - n_layer: تعداد تکرارهای message passing
    - cutoff_l: شعاع cutoff برای graph محلی، یعنی graph مبتنی بر bond/edge_index
    - cutoff_g: شعاع cutoff برای graph سراسری، یعنی اتصال اتم‌های نزدیک در فضای سه‌بعدی
    - use_weight_sharing: اگر True باشد یک لایه‌ی local/global چند بار تکرار می‌شود
    - use_basis_reduction: اگر True باشد تعداد RBF/SBF کمتر می‌شود
    """

    def __init__(
        self,
        dataset: str,
        dim: int,
        n_layer: int,
        cutoff_l: float,
        cutoff_g: float,
        flow: str = "source_to_target",
        use_weight_sharing: bool = False,
        use_basis_reduction: bool = False,
        num_spherical: int = 7,
        num_radial: int = 6,
        num_rbf: int = 16,
        reduced_num_spherical: int = 3, # it was 3
        reduced_num_radial: int = 4,
        reduced_num_rbf: int = 8, # 8
    ) -> None:
        # نام دیتاست برای سازگاری با ساختار پروژه نگه داشته می‌شود.
        self.dataset = dataset

        # ابعاد hidden vectorها و تعداد لایه‌های message passing.
        self.dim = dim
        self.n_layer = n_layer

        # cutoffهای local و global.
        self.cutoff_l = cutoff_l
        self.cutoff_g = cutoff_g

        # جهت ارسال پیام در PyTorch Geometric/layers.py.
        self.flow = flow

        # flagهای مربوط به تغییرات معماری.
        self.use_weight_sharing = use_weight_sharing
        self.use_basis_reduction = use_basis_reduction

        # اگر basis reduction فعال باشد، مقدارهای reduced استفاده می‌شوند.
        # در غیر این صورت همان مقدارهای original مدل استفاده می‌شوند.
        if use_basis_reduction:
            self.num_spherical = reduced_num_spherical
            self.num_radial = reduced_num_radial
            self.num_rbf = reduced_num_rbf
        else:
            self.num_spherical = num_spherical
            self.num_radial = num_radial
            self.num_rbf = num_rbf


class PAMNet(nn.Module):
    """
    نسخه‌ی کامل PAMNet.

    این مدل هم پیام‌های local و هم پیام‌های global را استفاده می‌کند.
    local graph از edge_index دیتاست می‌آید و معمولاً اتصال‌های bond را نشان می‌دهد.
    global graph با radius ساخته می‌شود و اتم‌هایی را که در فاصله‌ی cutoff_g هستند به هم وصل می‌کند.
    """

    def __init__(self, config: Config, envelope_exponent: int = 5):
        """
        سازنده‌ی PAMNet.

        envelope_exponent در Bessel/Spherical basis استفاده می‌شود تا اثر فاصله نزدیک cutoff نرم‌تر صفر شود.
        """
        super(PAMNet, self).__init__()

        # ذخیره‌ی تنظیمات اصلی روی object مدل.
        self.dataset = config.dataset
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff_l = config.cutoff_l
        self.cutoff_g = config.cutoff_g

        # این flagها از main_qm9.py و Config می‌آیند و تعیین می‌کنند کدام variant فعال باشد.
        self.use_weight_sharing = config.use_weight_sharing
        self.use_basis_reduction = config.use_basis_reduction

        # تعداد basis functionهایی که واقعاً استفاده می‌شود.
        # اگر br/wsbr انتخاب شده باشد، این مقدارها reduced هستند.
        self.num_spherical = config.num_spherical
        self.num_radial = config.num_radial
        self.num_rbf = config.num_rbf

        # embedding قابل‌آموزش برای نوع اتم‌ها در QM9.
        # QM9 در این پیاده‌سازی 5 نوع اتم اصلی دارد: H, C, N, O, F.
        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        # RBF برای فاصله‌های graph سراسری و graph محلی.
        # num_rbf در حالت original برابر 16 و در حالت basis reduction برابر 8 است.
        self.rbf_g = BesselBasisLayer(self.num_rbf, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(self.num_rbf, self.cutoff_l, envelope_exponent)

        # SBF برای encode کردن زاویه‌ها در message passing محلی.
        # در حالت original ابعاد آن 7*6 و در حالت reduced برابر 3*4 است.
        self.sbf = SphericalBasisLayer(
            self.num_spherical,
            self.num_radial,
            self.cutoff_l,
            envelope_exponent,
        )

        # MLPها basisهای فاصله/زاویه را به hidden dimension مدل تبدیل می‌کنند.
        self.mlp_rbf_g = MLP([self.num_rbf, self.dim])
        self.mlp_rbf_l = MLP([self.num_rbf, self.dim])
        self.mlp_sbf1 = MLP([self.num_spherical * self.num_radial, self.dim])
        self.mlp_sbf2 = MLP([self.num_spherical * self.num_radial, self.dim])

        # در حالت weight sharing، فقط یک global layer و یک local layer ساخته می‌شود
        # و همان لایه‌ها n_layer بار در forward تکرار می‌شوند.
        # این کار تعداد پارامترها را کم می‌کند.
        if self.use_weight_sharing:
            self.global_layer = Global_MessagePassing(config)
            self.local_layer = Local_MessagePassing(config)
        else:
            # در حالت original، برای هر عمق یک لایه‌ی جداگانه با وزن‌های جدا ساخته می‌شود.
            self.global_layer = nn.ModuleList(
                [Global_MessagePassing(config) for _ in range(config.n_layer)]
            )
            self.local_layer = nn.ModuleList(
                [Local_MessagePassing(config) for _ in range(config.n_layer)]
            )

        # softmax برای attention-based fusion بین خروجی‌های local و global استفاده می‌شود.
        self.softmax = nn.Softmax(dim=-1)

        # مقداردهی اولیه‌ی embeddingهای اتم.
        self.init()

    def init(self):
        """
        مقداردهی اولیه‌ی embedding اتم‌ها.

        مقدارها از توزیع uniform در بازه‌ی [-sqrt(3), sqrt(3)] گرفته می‌شوند.
        """
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def get_edge_info(self, edge_index, pos):
        """
        حذف self-loopها و محاسبه‌ی فاصله‌ی Euclidean برای هر edge.

        ورودی:
        - edge_index: ماتریس 2×E شامل source و target edgeها
        - pos: مختصات سه‌بعدی همه‌ی اتم‌ها

        خروجی:
        - edge_index بدون self-loop
        - dist: بردار طول E شامل فاصله‌ی هر دو اتم متصل
        """
        # self-loop یعنی edge از یک node به خودش؛ برای محاسبه‌ی فاصله/پیام اینجا لازم نیست.
        edge_index, _ = remove_self_loops(edge_index)

        # در این کد j منبع و i مقصد edge در نظر گرفته شده‌اند.
        j, i = edge_index

        # فاصله‌ی Euclidean بین pos[i] و pos[j].
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        return edge_index, dist

    def indices(self, edge_index, num_nodes):
        """
        ساخت indexهای لازم برای محاسبه‌ی زاویه‌ها در graph محلی.

        PAMNet برای پیام‌های محلی فقط فاصله‌ی دو اتم را استفاده نمی‌کند؛ زاویه‌های سه‌اتمی را هم encode می‌کند.
        برای این کار باید از edge_index ترکیب‌های i-j-k و pairهای مرتبط استخراج شوند.

        خروجی‌ها برای Local_MessagePassing و SphericalBasisLayer استفاده می‌شوند.
        """
        row, col = edge_index

        # value شماره‌ی هر edge است تا بعداً بتوانیم edge متناظر با هر triplet را پیدا کنیم.
        value = torch.arange(row.size(0), device=row.device)

        # adjacency ترانهاده‌شده به شکل sparse ساخته می‌شود تا همسایه‌ها سریع استخراج شوند.
        adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))

        # برای هر edge، همسایه‌های node مبدأ/مقصد استخراج می‌شوند تا triplet بسازیم.
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # idx_i, idx_j, idx_k سه node مربوط به زاویه‌ی دو-hop را مشخص می‌کنند.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # tripletهایی که i و k یک node هستند حذف می‌شوند، چون زاویه‌ی معتبر نمی‌سازند.
        mask = idx_i != idx_k
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # index edgeهای متناظر با k->j و j->i.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        # ساخت pairهای یک-hop برای زاویه‌ی نوع دوم.
        adj_t_col = adj_t[col]
        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()

        # pairهایی که j1 و j2 برابرند حذف می‌شوند.
        mask_j = idx_j1_pair != idx_j2_pair
        idx_i_pair, idx_j1_pair, idx_j2_pair = (
            idx_i_pair[mask_j],
            idx_j1_pair[mask_j],
            idx_j2_pair[mask_j],
        )

        # index edgeهای متناظر برای pairها.
        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]

        return (
            idx_i,
            idx_j,
            idx_k,
            idx_kj,
            idx_ji,
            idx_i_pair,
            idx_j1_pair,
            idx_j2_pair,
            idx_jj_pair,
            idx_ji_pair,
        )

    def forward(self, data):
        """
        forward pass مدل برای یک batch از moleculeهای QM9.

        data باید این attributeها را داشته باشد:
        - x: نوع اتم‌ها به شکل index
        - pos: مختصات سه‌بعدی اتم‌ها
        - edge_index: اتصال‌های local یا bond graph
        - batch: مشخص می‌کند هر node متعلق به کدام molecule در batch است
        """
        # دریافت ورودی‌های اصلی از batch.
        x_raw = data.x
        batch = data.batch
        pos = data.pos
        edge_index_l = data.edge_index

        # تبدیل index نوع اتم به embedding قابل‌آموزش.
        x = torch.index_select(self.embeddings, 0, x_raw.long())

        # ساخت graph سراسری: هر دو اتم که فاصله‌شان کمتر از cutoff_g باشد به هم وصل می‌شوند.
        row, col = radius(pos, pos, self.cutoff_g, batch, batch, max_num_neighbors=1000)
        edge_index_g = torch.stack([row, col], dim=0)

        # محاسبه‌ی فاصله‌ی edgeهای graph سراسری.
        edge_index_g, dist_g = self.get_edge_info(edge_index_g, pos)

        # محاسبه‌ی فاصله‌ی edgeهای graph محلی که از دیتاست آمده است.
        edge_index_l, dist_l = self.get_edge_info(edge_index_l, pos)

        # ساخت indexهای لازم برای زاویه‌ها و message passing محلی.
        (
            idx_i,
            idx_j,
            idx_k,
            idx_kj,
            idx_ji,
            idx_i_pair,
            idx_j1_pair,
            idx_j2_pair,
            idx_jj_pair,
            idx_ji_pair,
        ) = self.indices(edge_index_l, num_nodes=x.size(0))

        # -----------------------------
        # زاویه‌ی دو-hop: i-j-k
        # -----------------------------
        pos_ji = pos[idx_j] - pos[idx_i]
        pos_kj = pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)  # dot product
        b = torch.cross(pos_ji, pos_kj, dim=-1).norm(dim=-1)  # norm(cross product)
        angle2 = torch.atan2(b, a)  # زاویه‌ی پایدارتر نسبت به arccos

        # -----------------------------
        # زاویه‌ی یک-hop/pair
        # -----------------------------
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair = pos_j1_pair - pos_i_pair
        pos_jj_pair = pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.cross(pos_ji_pair, pos_jj_pair, dim=-1).norm(dim=-1)
        angle1 = torch.atan2(b, a)

        # تبدیل فاصله‌ها به radial basis و زاویه‌ها به spherical basis.
        rbf_l = self.rbf_l(dist_l)
        rbf_g = self.rbf_g(dist_g)
        sbf1 = self.sbf(dist_l, angle1, idx_jj_pair)
        sbf2 = self.sbf(dist_l, angle2, idx_kj)

        # projection basisها به dim مدل.
        edge_attr_rbf_l = self.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = self.mlp_rbf_g(rbf_g)
        edge_attr_sbf1 = self.mlp_sbf1(sbf1)
        edge_attr_sbf2 = self.mlp_sbf2(sbf2)

        # لیست‌ها برای نگهداری خروجی هر لایه و attention score مربوط به آن.
        out_global = []
        out_local = []
        att_score_global = []
        att_score_local = []

        # اجرای n_layer مرحله message passing.
        for layer_idx in range(self.n_layer):
            # اگر weight sharing فعال باشد، همیشه همان object لایه استفاده می‌شود.
            # اگر فعال نباشد، از لایه‌ی جداگانه‌ی همان depth استفاده می‌شود.
            global_layer = self.global_layer if self.use_weight_sharing else self.global_layer[layer_idx]
            local_layer = self.local_layer if self.use_weight_sharing else self.local_layer[layer_idx]

            # پیام‌رسانی global براساس graph ساخته‌شده با radius.
            x, out_g, att_score_g = global_layer(x, edge_attr_rbf_g, edge_index_g)
            out_global.append(out_g)
            att_score_global.append(att_score_g)

            # پیام‌رسانی local براساس bond graph و اطلاعات زاویه‌ای.
            x, out_l, att_score_l = local_layer(
                x,
                edge_attr_rbf_l,
                edge_attr_sbf2,
                edge_attr_sbf1,
                idx_kj,
                idx_ji,
                idx_jj_pair,
                idx_ji_pair,
                edge_index_l,
            )
            out_local.append(out_l)
            att_score_local.append(att_score_l)

        # ترکیب attention scoreهای global و local.
        att_score = torch.cat((torch.cat(att_score_global, 0), torch.cat(att_score_local, 0)), dim=-1)
        att_score = F.leaky_relu(att_score, negative_slope=0.2)
        att_weight = self.softmax(att_score)

        # ترکیب خروجی‌های global/local با وزن attention.
        out = torch.cat((torch.cat(out_global, 0), torch.cat(out_local, 0)), dim=-1)
        out = (out * att_weight).sum(dim=-1)

        # تبدیل خروجی ترکیب‌شده به مقدار node-level scalar.
        out = out.sum(dim=0).unsqueeze(-1)

        # aggregation از node-level به graph-level؛ هر molecule یک خروجی نهایی می‌گیرد.
        out = global_add_pool(out, batch)

        # خروجی نهایی به شکل بردار یک‌بعدی با طول batch_size.
        return out.view(-1)


class PAMNet_s(nn.Module):
    """
    نسخه‌ی ساده‌تر PAMNet.

    تفاوت اصلی با PAMNet کامل این است که Local_MessagePassing_s ساده‌تر است و فقط یک نوع spherical basis/angle را استفاده می‌کند.
    باقی منطق کلی، یعنی global graph، local graph، attention fusion و pooling مشابه است.
    """

    def __init__(self, config: Config, envelope_exponent: int = 5):
        """سازنده‌ی PAMNet_s با همان flagهای معماری Config."""
        super(PAMNet_s, self).__init__()

        # ذخیره‌ی تنظیمات معماری.
        self.dataset = config.dataset
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff_l = config.cutoff_l
        self.cutoff_g = config.cutoff_g
        self.use_weight_sharing = config.use_weight_sharing
        self.use_basis_reduction = config.use_basis_reduction
        self.num_spherical = config.num_spherical
        self.num_radial = config.num_radial
        self.num_rbf = config.num_rbf

        # embedding نوع اتم‌ها.
        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        # basisهای فاصله و زاویه.
        self.rbf_g = BesselBasisLayer(self.num_rbf, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(self.num_rbf, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            self.num_spherical,
            self.num_radial,
            self.cutoff_l,
            envelope_exponent,
        )

        # projection basisها به hidden dimension.
        self.mlp_rbf_g = MLP([self.num_rbf, self.dim])
        self.mlp_rbf_l = MLP([self.num_rbf, self.dim])
        self.mlp_sbf = MLP([self.num_spherical * self.num_radial, self.dim])

        # ساخت لایه‌ها با یا بدون weight sharing.
        if self.use_weight_sharing:
            self.global_layer = Global_MessagePassing(config)
            self.local_layer = Local_MessagePassing_s(config)
        else:
            self.global_layer = nn.ModuleList(
                [Global_MessagePassing(config) for _ in range(config.n_layer)]
            )
            self.local_layer = nn.ModuleList(
                [Local_MessagePassing_s(config) for _ in range(config.n_layer)]
            )

        # softmax برای attention fusion.
        self.softmax = nn.Softmax(dim=-1)
        self.init()

    def init(self):
        """مقداردهی اولیه‌ی embedding اتم‌ها."""
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def indices(self, edge_index, num_nodes):
        """
        ساخت indexهای pair برای نسخه‌ی ساده‌ی local message passing.

        در PAMNet_s فقط pairهای لازم برای یک نوع زاویه ساخته می‌شوند، بنابراین خروجی این تابع کمتر از PAMNet کامل است.
        """
        row, col = edge_index

        # شماره‌گذاری edgeها برای استفاده در spherical basis.
        value = torch.arange(row.size(0), device=row.device)

        # ساخت adjacency sparse.
        adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
        adj_t_col = adj_t[col]

        # تعداد همسایه‌های لازم برای هر edge.
        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)

        # indexهای سه node مربوط به pair/angle.
        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()

        # حذف pairهای نامعتبر که دو node میانی برابر می‌شوند.
        mask_j = idx_j1_pair != idx_j2_pair
        idx_i_pair, idx_j1_pair, idx_j2_pair = (
            idx_i_pair[mask_j],
            idx_j1_pair[mask_j],
            idx_j2_pair[mask_j],
        )

        # index edgeهای متناظر با pairها.
        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]

        return idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair

    def forward(self, data):
        """
        forward pass نسخه‌ی PAMNet_s برای یک batch از moleculeها.
        """
        # ورودی‌های graph batch.
        x_raw = data.x
        edge_index_l = data.edge_index
        pos = data.pos
        batch = data.batch

        # تبدیل نوع اتم به embedding.
        x = torch.index_select(self.embeddings, 0, x_raw.long())

        # -----------------------------
        # graph محلی و فاصله‌های local
        # -----------------------------
        edge_index_l, _ = remove_self_loops(edge_index_l)
        j_l, i_l = edge_index_l
        dist_l = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()

        # -----------------------------
        # graph سراسری و فاصله‌های global
        # -----------------------------
        row, col = radius(pos, pos, self.cutoff_g, batch, batch, max_num_neighbors=500)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        # indexهای لازم برای محاسبه‌ی زاویه‌ها.
        idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = self.indices(
            edge_index_l, num_nodes=x.size(0)
        )

        # محاسبه‌ی زاویه‌ی local برای spherical basis.
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair = pos_j1_pair - pos_i_pair
        pos_jj_pair_vec = pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair_vec).sum(dim=-1)
        b = torch.cross(pos_ji_pair, pos_jj_pair_vec, dim=-1).norm(dim=-1)
        angle = torch.atan2(b, a)

        # basis embedding برای فاصله‌ها و زاویه.
        rbf_l = self.rbf_l(dist_l)
        rbf_g = self.rbf_g(dist_g)
        sbf = self.sbf(dist_l, angle, idx_jj_pair)

        # projection basisها به hidden dimension.
        edge_attr_rbf_l = self.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = self.mlp_rbf_g(rbf_g)
        edge_attr_sbf = self.mlp_sbf(sbf)

        # لیست خروجی‌های هر layer برای fusion نهایی.
        out_global = []
        out_local = []
        att_score_global = []
        att_score_local = []

        # اجرای n_layer مرحله message passing با یا بدون weight sharing.
        for layer_idx in range(self.n_layer):
            global_layer = self.global_layer if self.use_weight_sharing else self.global_layer[layer_idx]
            local_layer = self.local_layer if self.use_weight_sharing else self.local_layer[layer_idx]

            # پیام‌رسانی global.
            x, out_g, att_score_g = global_layer(x, edge_attr_rbf_g, edge_index_g)
            out_global.append(out_g)
            att_score_global.append(att_score_g)

            # پیام‌رسانی local ساده‌شده.
            x, out_l, att_score_l = local_layer(
                x,
                edge_attr_rbf_l,
                edge_attr_sbf,
                idx_jj_pair,
                idx_ji_pair,
                edge_index_l,
            )
            out_local.append(out_l)
            att_score_local.append(att_score_l)

        # attention fusion بین خروجی‌های global و local.
        att_score = torch.cat((torch.cat(att_score_global, dim=0), torch.cat(att_score_local, dim=0)), dim=-1)
        att_score = F.leaky_relu(att_score, negative_slope=0.2)
        att_weight = self.softmax(att_score)

        # اعمال attention weight روی خروجی‌ها.
        out = torch.cat((torch.cat(out_global, dim=0), torch.cat(out_local, dim=0)), dim=-1)
        out = (out * att_weight).sum(dim=-1)

        # تبدیل خروجی node-level به scalar و سپس pooling به graph-level.
        out = out.sum(dim=0).unsqueeze(-1)
        out = global_add_pool(out, batch)

        # خروجی نهایی برای هر molecule در batch.
        return out.view(-1)
