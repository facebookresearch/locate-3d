# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import torch
from transformers import AutoTokenizer
from transformers import CLIPTextModelWithProjection
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from models.model_utils.bbox_utils import box_cxcyczwhd_to_xyzxyz_jit


class LearnedPosEmbeddings(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, dim=3, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(dim, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1),
        )

    def forward(self, xyz, train_3d=False):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding.transpose(1, 2)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class Block(nn.Module):
    def __init__(
        self, d_model, n_heads, dim_feedforward, dropout, drop_path, is_self_attn
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.query_norm = nn.LayerNorm(d_model)
        self.keys_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.is_self_attn = is_self_attn
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries, queries_pos_embed, keys, keys_pos_embed, mask):
        if self.is_self_attn:
            normed_queries = self.query_norm(queries)
            normed_keys = self.query_norm(keys)
        else:
            normed_queries = self.query_norm(queries)
            normed_keys = keys

        attn_out = self.attn(
            (normed_queries + queries_pos_embed).transpose(0, 1),
            (normed_keys + keys_pos_embed).transpose(0, 1),
            (normed_keys + keys_pos_embed).transpose(0, 1),
            key_padding_mask=mask,
            need_weights=False,
        )[0].transpose(0, 1)

        queries = queries + self.drop_path(self.dropout(attn_out))
        queries = queries + self.drop_path(self.ffn(self.ffn_norm(queries)))

        return queries


class TransformerModule(nn.Module):
    def __init__(
        self, d_model, n_heads, dim_feedforward, dropout, drop_path, use_checkpointing
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.query_self_attn = Block(
            d_model, n_heads, dim_feedforward, dropout, drop_path, is_self_attn=True
        )
        self.query_ptc_feat_attn = Block(
            d_model, n_heads, dim_feedforward, dropout, drop_path, is_self_attn=False
        )
        self.ptc_feat_query_attn = Block(
            d_model, n_heads, dim_feedforward, dropout, drop_path, is_self_attn=False
        )

    def maybe_checkpoint(self, func, *args):
        if self.use_checkpointing:
            return checkpoint(func, *args, use_reentrant=False)
        return func(*args)

    def forward(
        self,
        query_feats,
        query_pos_embed,
        text_feats,
        text_pos_embed,
        ptc_feats,
        ptc_pos_embed,
        query_mask,
        ptc_mask,
    ):
        query_feats = torch.cat([query_feats, text_feats], dim=1)
        query_pos_embed = torch.cat([query_pos_embed, text_pos_embed], dim=1)

        query_feats = self.maybe_checkpoint(
            self.query_self_attn,
            query_feats,
            query_pos_embed,
            query_feats,
            query_pos_embed,
            query_mask,
        )
        query_feats = self.maybe_checkpoint(
            self.query_ptc_feat_attn,
            query_feats,
            query_pos_embed,
            ptc_feats,
            ptc_pos_embed,
            ptc_mask,
        )
        ptc_feats = self.maybe_checkpoint(
            self.ptc_feat_query_attn,
            ptc_feats,
            ptc_pos_embed,
            query_feats,
            query_pos_embed,
            query_mask,
        )

        text_feats = query_feats[:, -text_feats.shape[1] :]
        query_feats = query_feats[:, : -text_feats.shape[1]]
        return query_feats, text_feats, ptc_feats


class MaskPredictionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.mask_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, query_feats, ptc_feats):
        query_feats = self.mask_embed(query_feats)
        return torch.einsum("bqc,bnc->bqn", query_feats, ptc_feats)


class TextAlignmentHead(nn.Module):
    def __init__(self, d_model, max_tokens):
        super().__init__()

        self.text_alignment_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(d_model, d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(d_model, max_tokens, 1),
        )

    def forward(self, query_feats):
        scores = self.text_alignment_head(query_feats.permute(0, 2, 1))
        return scores.transpose(1, 2)


class BBoxHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query_projector = nn.Linear(d_model, d_model * 2)
        self.xyz_projector = nn.Linear(3, d_model)
        self.feature_projector = nn.Identity()

        self.cross_attention = nn.MultiheadAttention(
            d_model * 2, num_heads=16, dropout=0.1, batch_first=True
        )

        self.bbox_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 6),
        )

    def net(self, query_feats, ptc_feats, ptc_xyz, ptc_mask=None):
        query_feats = self.query_projector(query_feats)
        ptc_xyz = self.xyz_projector(ptc_xyz)
        ptc_feats = self.feature_projector(ptc_feats)

        ptc_feats_with_xyz = torch.cat([ptc_feats, ptc_xyz], dim=-1)
        attended_features = self.cross_attention(
            query_feats,
            ptc_feats_with_xyz,
            ptc_feats_with_xyz,
            key_padding_mask=ptc_mask,
            need_weights=False,
        )[0]

        bbox_pred = self.bbox_predictor(attended_features)
        center = bbox_pred[..., :3]
        dimensions = F.softplus(bbox_pred[..., 3:])

        return torch.cat([center, dimensions], dim=-1)

    def forward(self, query_feats, ptc_feats, ptc_xyz, ptc_mask=None):
        return checkpoint(
            self.net, query_feats, ptc_feats, ptc_xyz, ptc_mask, use_reentrant=False
        )


class Locate3DDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        input_feat_dim,
        num_queries,
        num_decoder_layers,
        transformer_n_heads,
        transformer_dim_feedforward,
        transformer_dropout,
        transformer_max_drop_path,
        transformer_use_checkpointing,
        freeze_text_encoder,
        text_encoder,
    ):
        super(Locate3DDecoder, self).__init__()
        self.num_decoder_layers = num_decoder_layers

        # Text Encoding Model
        assert text_encoder in ["clip", "clip-large"], "Only CLIP models are supported"
        self.clip_model = "openai/clip-vit-large-patch14"
        self.tokenizer = AutoTokenizer.from_pretrained(self.clip_model)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(self.clip_model)
        self.text_encoder_hidden_size = self.text_encoder.config.hidden_size
        self.max_tokens = 77

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Text and ptc feature projectors before they go into the transformer
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder_hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1),
        )
        self.ptc_feat_projector = nn.Sequential(
            nn.Linear(input_feat_dim, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1),
        )

        # Positional Embeddings
        self.pos_embed_3d = LearnedPosEmbeddings(dim=3, num_pos_feats=d_model)

        # Learned Queries
        self.query_feat = nn.Embedding(num_queries, d_model)
        self.query_pos = nn.Embedding(num_queries, d_model)

        # Transformer
        self.decoder = nn.ModuleList()
        drop_paths = [
            x.item()
            for x in torch.linspace(0, transformer_max_drop_path, num_decoder_layers)
        ]
        for i in range(num_decoder_layers):
            self.decoder.append(
                TransformerModule(
                    d_model,
                    transformer_n_heads,
                    transformer_dim_feedforward,
                    transformer_dropout,
                    drop_paths[i],
                    transformer_use_checkpointing,
                )
            )

        # Mask Prediction Head
        self.mask_prediction_heads = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.mask_prediction_heads.append(MaskPredictionHead(d_model))

        # Text Alignment Head
        self.text_alignment_head = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.text_alignment_head.append(
                TextAlignmentHead(d_model, max_tokens=self.max_tokens)
            )

        # Bbox Regression Head
        self.bbox_head = BBoxHead(d_model)

        # Init Params
        self._init_bn_momentum()

    def _init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1

    def forward(self, encoded_scene, query):
        """Forward pass for the entire model."""
        # Prepare inputs
        pointclouds = [encoded_scene]
        captions = [query]
        ptc_feats = pad_sequence(
            [pcd["features"] for pcd in pointclouds], batch_first=True
        ).cuda()
        ptc_xyz = pad_sequence(
            [pcd["points"] for pcd in pointclouds], batch_first=True
        ).cuda()
        ptc_mask, query_mask = None, None

        # Outputs
        predictions_class = []
        predictions_mask = []
        predictions_boxes = []

        # Project ptc features and get positional embeddings
        ptc_feats = self.ptc_feat_projector(ptc_feats)
        ptc_pos = self.pos_embed_3d(ptc_xyz)

        # Tokenize text utterances
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            padding="longest",
            return_tensors="pt",
            max_length=self.max_tokens,
            truncation=True,
        ).to(ptc_feats.device)
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        # Encode text utterances
        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state)
        text_pos = torch.zeros_like(text_feats)

        # Learned queries
        batch_size = len(text_feats)
        query_feats = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_pos = self.query_pos.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # Decoder
        for i in range(self.num_decoder_layers):

            # Transformer
            query_feats, text_feats, ptc_feats = self.decoder[i](
                query_feats,
                query_pos,
                text_feats,
                text_pos,
                ptc_feats,
                ptc_pos,
                query_mask,
                ptc_mask,
            )

            # Prediction Heads
            mask = self.mask_prediction_heads[i](query_feats, ptc_feats)
            text_alignment = self.text_alignment_head[i](query_feats)
            bbox = self.bbox_head(query_feats, ptc_feats, ptc_xyz, ptc_mask=ptc_mask)
            bbox = box_cxcyczwhd_to_xyzxyz_jit(bbox)

            # Store predictions for deep supervision
            predictions_class.append(text_alignment)
            predictions_mask.append(mask)
            predictions_boxes.append(bbox)

        # Prepare outputs
        out = {
            "text_attn_mask": text_attention_mask,
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_boxes": predictions_boxes[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class, predictions_mask, predictions_boxes
            ),
        }

        return out

    @torch.jit.unused
    def _set_aux_loss(self, out_classes, out_masks, out_boxes):
        """Auxiliary loss computation."""
        aux_loss_output = []
        for a, b, c in zip(out_classes[:-1], out_masks[:-1], out_boxes[:-1]):
            aux_loss_output.append(
                {
                    "pred_logits": a,
                    "pred_masks": b,
                    "pred_boxes": c,
                }
            )
        return aux_loss_output
