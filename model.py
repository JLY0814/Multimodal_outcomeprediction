"""
Dual-branch 3-D CNN with a DLA-inspired backbone.

Architecture (single-instance mode, use_mil=False)
────────────
  CT  input (1, D, H, W)  ──► DLA3D backbone ──► GAP ──► embed_ct  (B, C)
  PET input (1, D, H, W)  ──► DLA3D backbone ──► GAP ──► embed_pet (B, C)
                                                               │
                                                          concat (B, 2C)
                                                               │
                                                     MLP (FC-BN-ReLU-Drop)
                                                               │

MIL-Attention mode (use_mil=True)
──────────────────────────────────
  CT  input (N, 1, D, H, W)  ──► DLA3D ──► GAP ──►╮
  PET input (N, 1, D, H, W)  ──► DLA3D ──► GAP ──►╡ concat per instance (N, 2C)
  [Dose input]                                      ╯
                                      GatedAttentionMIL ──► patient emb (2C)
                                                               │
                                                     MLP ──► logit
                                                     FC → sigmoid → prob

DLA backbone (Deep Layer Aggregation, adapted to 3-D):
  Stage 0: stem conv  → out0
  Stage 1: basic blocks → out1
  Stage 2: basic blocks → out2  (+ IDA from out1)
  Stage 3: basic blocks → out3  (+ IDA from out2)
  Stage 4: basic blocks → out4  (+ IDA from out3)
  Global Average Pool → (B, channels[4])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBnRelu3d(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, bias=False):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel, stride, padding, bias=bias),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


class BasicBlock3D(nn.Module):
    """3-D residual block (two 3×3×3 convs)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


def make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int = 1) -> nn.Sequential:
    layers = [BasicBlock3D(in_ch, out_ch, stride=stride)]
    for _ in range(1, n_blocks):
        layers.append(BasicBlock3D(out_ch, out_ch))
    return nn.Sequential(*layers)


class IDAUp3D(nn.Module):
    """
    Iterative Deep Aggregation up-node:
    fuses a lower-resolution feature (up-sampled) with a same-resolution feature.
    """

    def __init__(self, in_ch_lo: int, in_ch_hi: int, out_ch: int):
        super().__init__()
        # project low-res features to out_ch
        self.proj  = ConvBnRelu3d(in_ch_lo, out_ch, kernel=1, padding=0)
        # project high-res features to out_ch
        self.node  = ConvBnRelu3d(in_ch_hi, out_ch, kernel=1, padding=0)
        # final aggregation
        self.agg   = ConvBnRelu3d(out_ch * 2, out_ch)

    def forward(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        # upsample lo to match hi spatial dims
        lo_up = F.interpolate(lo, size=hi.shape[2:], mode="trilinear", align_corners=False)
        lo_up = self.proj(lo_up)
        hi    = self.node(hi)
        return self.agg(torch.cat([lo_up, hi], dim=1))


# ── DLA-inspired 3-D backbone ─────────────────────────────────────────────────

class DLA3D(nn.Module):
    """
    3-D DLA backbone with 5 stages.

    channels: list of output channels per stage [stem, s1, s2, s3, s4].
    blocks:   number of BasicBlock3D per stage.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = None,
        blocks: List[int] = None,
    ):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128, 256]
        if blocks is None:
            blocks = [1, 1, 2, 2, 2]

        assert len(channels) == 5 and len(blocks) == 5

        # Stage 0 – stem (no downsampling)
        self.stem = nn.Sequential(
            ConvBnRelu3d(in_channels, channels[0], kernel=7, stride=1, padding=3),
            ConvBnRelu3d(channels[0], channels[0]),
        )

        # Stage 1 – first downsampling (stride 2 in H, W; stride 1 in D to preserve slices)
        self.stage1 = make_stage(channels[0], channels[1], blocks[1],
                                 stride=(1, 2, 2))

        # Stage 2
        self.stage2 = make_stage(channels[1], channels[2], blocks[2],
                                 stride=(1, 2, 2))
        self.ida2   = IDAUp3D(channels[1], channels[2], channels[2])

        # Stage 3
        self.stage3 = make_stage(channels[2], channels[3], blocks[3],
                                 stride=(1, 2, 2))
        self.ida3   = IDAUp3D(channels[2], channels[3], channels[3])

        # Stage 4
        self.stage4 = make_stage(channels[3], channels[4], blocks[4],
                                 stride=(1, 2, 2))
        self.ida4   = IDAUp3D(channels[3], channels[4], channels[4])

        self.out_channels = channels[4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)          # (B, C0, D, H, W)
        s1 = self.stage1(s0)       # (B, C1, D, H/2, W/2)
        s2 = self.stage2(s1)       # (B, C2, D, H/4, W/4)
        s2 = self.ida2(s1, s2)     # IDA: fuse s1 into s2
        s3 = self.stage3(s2)       # (B, C3, D, H/8, W/8)
        s3 = self.ida3(s2, s3)     # IDA: fuse s2 into s3
        s4 = self.stage4(s3)       # (B, C4, D, H/16, W/16)
        s4 = self.ida4(s3, s4)     # IDA: fuse s3 into s4
        return s4


# ── Gated Attention MIL pooling ───────────────────────────────────────────────

class GatedAttentionMIL(nn.Module):
    """
    Gated attention-based MIL pooling (Ilse et al. NeurIPS 2018).

    Input:  H  (B, N, D)  — B patients, N instances, D-dim embeddings
    Output: patient_emb (B, D), attention_weights (B, N, 1)

    A = softmax( w · (tanh(V·H) ⊙ sigmoid(U·H)) )
    M = Σ_n  A_n · H_n
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden_dim)   # value branch
        self.U = nn.Linear(in_dim, hidden_dim)   # gate branch
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H: torch.Tensor):
        # H: (B, N, D)
        gate  = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))  # (B, N, hidden)
        A     = self.w(gate)                                       # (B, N, 1)
        A     = torch.softmax(A, dim=1)                            # normalise over N
        M     = (A * H).sum(dim=1)                                 # (B, D)
        return M, A


# ── Dual/Tri-branch model ──────────────────────────────────────────────────────

class DualBranch3DCNN(nn.Module):
    """
    Dual-branch (CT + PET) or tri-branch (CT + PET + Dose) 3-D CNN.

    Args:
        dla_channels:   channel list for CT and PET DLA backbones.
        fusion_hidden:  MLP hidden dimension after concat.
        dropout:        dropout probability.
        num_classes:    1 for binary sigmoid output.
        use_dose:       if True, add a third Dose branch (late fusion).
        dose_channels:  channel list for the (shallower) Dose backbone.
        use_mil:        if True, use Gated-Attention MIL pooling over N instances.
                        Input shape becomes (B, N, 1, D, H, W) per modality.
        mil_attn_dim:   hidden dim of the gated attention network.
        ct_only:        if True, use only the CT branch (ablation study).
                        PET and Dose branches are disabled; fused_dim = dla_channels[-1].
        pet_only:       if True, use only the PET branch (ablation study).
                        CT and Dose branches are disabled; fused_dim = dla_channels[-1].
                        Mutually exclusive with ct_only and use_missing_gate.
    """

    def __init__(
        self,
        dla_channels:      List[int] = None,
        fusion_hidden:     int  = 256,
        dropout:           float = 0.5,
        num_classes:       int  = 1,
        use_dose:          bool = False,
        dose_channels:     List[int] = None,
        use_mil:           bool = False,
        mil_attn_dim:      int  = 128,
        ct_only:           bool = False,
        pet_only:          bool = False,
        use_missing_gate:  bool = False,
        gate_hidden:       int  = 64,
        use_unipair:       bool = False,
    ):
        super().__init__()
        if dla_channels is None:
            dla_channels = [16, 32, 64, 128, 256]
        if dose_channels is None:
            dose_channels = [8, 16, 32, 64, 128]

        assert not (ct_only and pet_only), "ct_only and pet_only are mutually exclusive."

        self.use_dose         = use_dose
        self.use_mil          = use_mil
        self.ct_only          = ct_only
        self.pet_only         = pet_only
        self.use_missing_gate = use_missing_gate
        self.use_unipair      = use_unipair

        embed_ct = dla_channels[-1]
        self.embed_ct = embed_ct   # stored for head_ct input slicing

        if pet_only:
            # Ablation: PET only – no CT, no Dose
            self.pet_branch = DLA3D(in_channels=1, channels=dla_channels)
            fused_dim = embed_ct
        elif ct_only:
            # Ablation: CT only – no PET, no Dose
            self.ct_branch = DLA3D(in_channels=1, channels=dla_channels)
            fused_dim = embed_ct
        else:
            # Dual/Tri-branch: CT + PET (+ optional Dose)
            self.ct_branch  = DLA3D(in_channels=1, channels=dla_channels)
            self.pet_branch = DLA3D(in_channels=1, channels=dla_channels)
            fused_dim = embed_ct * 2

            # Missing-aware gate: g = sigmoid(MLP([z_ct, pet_present])) → scales z_pet
            if use_missing_gate:
                self.pet_gate = nn.Sequential(
                    nn.Linear(embed_ct + 1, gate_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(gate_hidden, 1),
                    nn.Sigmoid(),
                )

            # Optional Dose branch (shallower backbone)
            if use_dose:
                self.dose_branch = DLA3D(in_channels=1, channels=dose_channels)
                fused_dim += dose_channels[-1]

        # Optional MIL-Attention pooling (operates on fused instance embeddings)
        if use_mil:
            self.mil_attention = GatedAttentionMIL(fused_dim, hidden_dim=mil_attn_dim)

        # Classifier head(s)
        # use_missing_gate=True → two conditional heads (one per modality group);
        # otherwise → single shared head.
        # LayerNorm is used throughout missing-gate mode because CT-only and paired
        # samples share a batch and BatchNorm1d statistics would be corrupted by the
        # structurally-zero PET dimensions in CT-only samples.
        def _make_head(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, fusion_hidden),
                nn.LayerNorm(fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                nn.LayerNorm(fusion_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden // 2, num_classes),
            )

        if use_missing_gate:
            # head_fused: paired patients  (input = full fused embedding)
            # head_ct:    CT-only patients (input = CT embedding only, first embed_ct dims)
            self.head_fused = _make_head(fused_dim)
            self.head_ct    = _make_head(embed_ct)
        elif use_unipair:
            # Unipair mode: separate symmetric head for each modality.
            # head_ct  receives CT-only embedding  (first embed_ct dims of fused)
            # head_pet receives PET-only embedding (second embed_ct dims of fused)
            # Both heads run on ALL samples so DDP parameters stay symmetric.
            self.head_ct = nn.Sequential(
                nn.LayerNorm(embed_ct),
                nn.Linear(embed_ct, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden, num_classes),
            )
            self.head_pet = nn.Sequential(
                nn.LayerNorm(embed_ct),
                nn.Linear(embed_ct, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden, num_classes),
            )
        else:
            _Norm1d = nn.BatchNorm1d
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, fusion_hidden),
                _Norm1d(fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                _Norm1d(fusion_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden // 2, num_classes),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _encode(
        self,
        ct:          torch.Tensor,     # (B, 1, D, H, W)
        pet:         torch.Tensor = None,
        dose:        torch.Tensor = None,
        pet_present: torch.Tensor = None,  # (B,) float 0/1; None → assume all present
    ) -> torch.Tensor:
        """Encode a batch of single-instance patches → fused embedding (B, fused_dim).
        When pet_only=True, the 'ct' argument carries PET data (mirrors ct_only pattern)."""
        if self.pet_only:
            # 'ct' argument holds PET tensor (see dataset/train pet_only 2-tuple)
            emb_pet = F.adaptive_avg_pool3d(self.pet_branch(ct), 1).flatten(1)
            return emb_pet
        emb_ct = F.adaptive_avg_pool3d(self.ct_branch(ct), 1).flatten(1)
        if self.ct_only:
            return emb_ct
        if self.use_missing_gate:
            # Determine which samples have real PET data.
            if pet_present is not None:
                pp     = pet_present.view(-1, 1).float()                      # (B, 1)
                paired = pp.squeeze(1).bool()                                  # (B,)
            else:
                pp     = emb_ct.new_ones(emb_ct.shape[0], 1)
                paired = torch.ones(emb_ct.shape[0], dtype=torch.bool,
                                    device=emb_ct.device)
            # Always run pet_branch on ALL inputs (DDP requires symmetric parameter
            # participation on every rank).  The hard mask (* pp) guarantees
            # emb_pet is exactly 0 for CT-only samples regardless of gate output.
            emb_pet = F.adaptive_avg_pool3d(self.pet_branch(pet), 1).flatten(1)
            g       = self.pet_gate(torch.cat([emb_ct, pp], dim=1))           # (B, 1)
            emb_pet = g * emb_pet * pp                                         # (B, C)
        else:
            emb_pet = F.adaptive_avg_pool3d(self.pet_branch(pet), 1).flatten(1)
        if self.use_dose and dose is not None:
            emb_dose = F.adaptive_avg_pool3d(self.dose_branch(dose), 1).flatten(1)
            return torch.cat([emb_ct, emb_pet, emb_dose], dim=1)
        return torch.cat([emb_ct, emb_pet], dim=1)

    def encode(
        self,
        ct:          torch.Tensor,
        pet:         torch.Tensor = None,
        dose:        torch.Tensor = None,
        pet_present: torch.Tensor = None,  # (B,) float 0/1
    ) -> torch.Tensor:
        """
        Return patient-level embedding (B, fused_dim), before the classifier head.

        MIL mode  (use_mil=True):  ct/pet/dose shape (B, N, 1, D, H, W)
        Single    (use_mil=False): ct/pet/dose shape (B, 1, D, H, W)
        ct_only=True: pet and dose are ignored.
        pet_present: (B,) tensor; 1=PET available, 0=CT-only. Used by missing gate.
        """
        if self.use_mil:
            B, N = ct.shape[:2]
            ct_in   = ct.view(B * N, *ct.shape[2:])
            pet_in  = pet.view(B * N, *pet.shape[2:]) if pet is not None else None
            dose_in = dose.view(B * N, *dose.shape[2:]) if dose is not None else None
            pp_in   = pet_present.repeat_interleave(N) if pet_present is not None else None
            emb = self._encode(ct_in, pet_in, dose_in, pp_in)  # (B*N, fused_dim)
            emb = emb.view(B, N, -1)                            # (B, N, fused_dim)
            patient_emb, _ = self.mil_attention(emb)            # (B, fused_dim)
            return patient_emb
        else:
            return self._encode(ct, pet, dose, pet_present)     # (B, fused_dim)

    def _apply_head(
        self,
        patient_emb:      torch.Tensor,       # (B, fused_dim)
        pet_present:      torch.Tensor = None, # (B,) float 0/1
        return_ct_logits: bool         = False,
        modality:         torch.Tensor = None, # (B,) float 0=CT / 1=PET; unipair mode
    ):
        """
        Route each sample to the correct classifier head and return logits (B,).

        use_missing_gate=True:
          - Both head_fused and head_ct run on ALL samples (DDP requires symmetric
            parameter use across ranks).
          - torch.where selects head_fused output for paired (pet_present=1) and
            head_ct output for CT-only (pet_present=0).
          - head_ct receives only the CT portion of the embedding (first embed_ct dims).

        use_unipair=True:
          - Both head_ct and head_pet run on ALL samples (DDP symmetric requirement).
          - head_ct  receives emb[:, :embed_ct]  (CT embedding portion).
          - head_pet receives emb[:, embed_ct:embed_ct*2] (PET embedding portion).
          - torch.where(modality.bool(), ...) selects the PET head for modality=1.

        use_missing_gate=False, use_unipair=False:
          - Single self.classifier as before.

        return_ct_logits=True (only meaningful when use_missing_gate=True):
          - Returns (logits, logits_ct) tuple instead of logits scalar.
          - logits_ct is the raw head_ct output on ALL patients (auxiliary CT loss).
          - When use_missing_gate=False, returns (logits, None).
        """
        if self.use_missing_gate:
            emb_ct_only  = patient_emb[:, :self.embed_ct]       # (B, embed_ct)
            logits_fused = self.head_fused(patient_emb).squeeze(1)   # (B,)
            logits_ct    = self.head_ct(emb_ct_only).squeeze(1)      # (B,)
            if pet_present is not None:
                pp = pet_present.bool()
                logits = torch.where(pp, logits_fused, logits_ct)
            else:
                logits = logits_fused   # fallback: assume all paired
            if return_ct_logits:
                return logits, logits_ct
            return logits
        if self.use_unipair and modality is not None:
            # Both heads run on all samples for DDP symmetry
            emb_ct_only  = patient_emb[:, :self.embed_ct]                        # (B, C)
            emb_pet_only = patient_emb[:, self.embed_ct:self.embed_ct * 2]       # (B, C)
            logits_ct_head  = self.head_ct(emb_ct_only).squeeze(1)               # (B,)
            logits_pet_head = self.head_pet(emb_pet_only).squeeze(1)             # (B,)
            mod = modality.bool()   # True = PET (modality=1), False = CT (modality=0)
            return torch.where(mod, logits_pet_head, logits_ct_head)
        logits = self.classifier(patient_emb).squeeze(1)
        if return_ct_logits:
            return logits, None
        return logits

    def forward(
        self,
        ct:               torch.Tensor,
        pet:              torch.Tensor = None,
        dose:             torch.Tensor = None,
        return_attn:      bool = False,
        pet_present:      torch.Tensor = None,  # (B,) float 0/1; None → assume all present
        return_ct_logits: bool = False,
        modality:         torch.Tensor = None,  # (B,) float 0=CT / 1=PET; use_unipair only
    ):
        """
        Single-instance (use_mil=False):
            ct, pet, dose : (B, 1, D, H, W)
            returns logits : (B,)

        MIL mode (use_mil=True):
            ct, pet, dose : (B, N, 1, D, H, W)   N = number of instances
            returns logits : (B,)  or  (logits, A) when return_attn=True

        CT-only (ct_only=True):
            pet and dose are ignored; pass only ct.
        PET-only (pet_only=True):
            pass pet tensor as the first positional argument (ct param).

        pet_present: (B,) tensor; 1=PET available, 0=CT-only. Enables missing gate.
        modality:    (B,) tensor; 0=CT sample, 1=PET sample. Enables unipair routing.
        return_attn: if True (MIL mode only), also return attention weights A (B, N, 1).
        return_ct_logits: if True and use_missing_gate=True, also return head_ct logits
            on ALL patients (for auxiliary CT loss). Combinations:
              return_attn=True,  return_ct_logits=True  → (logits, A, logits_ct)
              return_attn=True,  return_ct_logits=False → (logits, A)
              return_attn=False, return_ct_logits=True  → (logits, logits_ct)
              default                                   → logits
        """
        if self.use_mil:
            B, N = ct.shape[:2]
            # Flatten B and N for efficient batch processing
            ct_in   = ct.view(B * N, *ct.shape[2:])
            pet_in  = pet.view(B * N, *pet.shape[2:]) if pet is not None else None
            dose_in = dose.view(B * N, *dose.shape[2:]) if dose is not None else None
            pp_in   = pet_present.repeat_interleave(N) if pet_present is not None else None

            emb = self._encode(ct_in, pet_in, dose_in, pp_in)  # (B*N, fused_dim)
            emb = emb.view(B, N, -1)                            # (B, N, fused_dim)

            patient_emb, A = self.mil_attention(emb)            # (B, fused_dim), (B, N, 1)
            head_out = self._apply_head(patient_emb, pet_present,
                                        return_ct_logits=return_ct_logits,
                                        modality=modality)
            if return_ct_logits:
                logits, logits_ct = head_out
                if return_attn:
                    return logits, A, logits_ct
                return logits, logits_ct
            logits = head_out
            if return_attn:
                return logits, A
            return logits

        else:
            fused    = self._encode(ct, pet, dose, pet_present)   # (B, fused_dim)
            head_out = self._apply_head(fused, pet_present,
                                        return_ct_logits=return_ct_logits,
                                        modality=modality)
            if return_ct_logits:
                return head_out   # (logits, logits_ct) tuple
            return head_out       # logits scalar


# ── Dual-scale branch (large patch) ───────────────────────────────────────────

class LargePatchBranch(nn.Module):
    """
    Large-patch branch: Dose_large (+ optionally CT_large) → embedding.

    Args:
        ct_channels:   channel list for the CT_large DLA3D backbone.
        dose_channels: channel list for the Dose_large DLA3D backbone.
        use_ct:        if False (default), only Dose_large is processed.
                       ct_large is still accepted in forward() but ignored.
    """

    def __init__(
        self,
        ct_channels:   List[int] = None,
        dose_channels: List[int] = None,
        use_ct:        bool      = False,
    ):
        super().__init__()
        if ct_channels is None:
            ct_channels = [8, 16, 32, 64, 128]
        if dose_channels is None:
            dose_channels = [8, 16, 32, 64, 128]

        self.use_ct      = use_ct
        self.dose_branch = DLA3D(in_channels=1, channels=dose_channels)
        if use_ct:
            self.ct_branch = DLA3D(in_channels=1, channels=ct_channels)
            self.out_dim = ct_channels[-1] + dose_channels[-1]
        else:
            self.out_dim = dose_channels[-1]

    def forward(self, ct_large: torch.Tensor, dose_large: torch.Tensor) -> torch.Tensor:
        """
        ct_large  : (B, 1, D_l, H_l, W_l)  – ignored when use_ct=False
        dose_large: (B, 1, D_l, H_l, W_l)
        returns   : (B, out_dim)
        """
        emb_dose = F.adaptive_avg_pool3d(self.dose_branch(dose_large), 1).flatten(1)
        if self.use_ct:
            emb_ct = F.adaptive_avg_pool3d(self.ct_branch(ct_large), 1).flatten(1)
            return torch.cat([emb_ct, emb_dose], dim=1)
        return emb_dose


class DualScaleModel(nn.Module):
    """
    Dual-scale model that fuses a small-patch branch (CT + PET) with a
    large-patch branch (CT_large + Dose_large).

    Branch 1: CT + PET small patch → DualBranch3DCNN.encode() → emb1
              Embedding dim: dla_channels[-1] * 2.
    Branch 2: Dose large patch (+ optionally CT large) → LargePatchBranch → emb2
              Embedding dim: dose_large_channels[-1]  (+ ct_large_channels[-1] if use_ct).
    Fusion:   concat(emb1, emb2) → MLP → logit.

    Args:
        dla_channels:             channel list for Branch-1 CT/PET DLA backbones.
        fusion_hidden:            MLP hidden dim inside Branch-1 (kept for compat).
        dropout:                  dropout probability.
        use_mil:                  if True, Branch-1 uses Gated-Attention MIL.
        mil_attn_dim:             hidden dim of gated attention.
        ct_large_channels:        channel list for Branch-2 CT_large backbone.
        dose_large_channels:      channel list for Branch-2 Dose_large backbone.
        dual_scale_fusion_hidden: hidden dim of the top-level fusion MLP.
        large_branch_use_ct:      if False (default), Branch-2 uses Dose only.
                                  if True, Branch-2 uses CT_large + Dose_large.
    """

    def __init__(
        self,
        dla_channels:              List[int] = None,
        fusion_hidden:             int       = 256,
        dropout:                   float     = 0.5,
        use_mil:                   bool      = False,
        mil_attn_dim:              int       = 128,
        ct_large_channels:         List[int] = None,
        dose_large_channels:       List[int] = None,
        dual_scale_fusion_hidden:  int       = 256,
        large_branch_use_ct:       bool      = True,
    ):
        super().__init__()
        if dla_channels is None:
            dla_channels = [16, 32, 64, 128, 256]
        if ct_large_channels is None:
            ct_large_channels = [8, 16, 32, 64, 128]
        if dose_large_channels is None:
            dose_large_channels = [8, 16, 32, 64, 128]

        # Branch 1: small-patch CT + PET (no dose; dose goes to large branch)
        self.branch1 = DualBranch3DCNN(
            dla_channels=dla_channels,
            fusion_hidden=fusion_hidden,
            dropout=dropout,
            use_dose=False,
            use_mil=use_mil,
            mil_attn_dim=mil_attn_dim,
        )

        # Branch 2: large-patch Dose (+ CT if large_branch_use_ct=True)
        self.branch2 = LargePatchBranch(
            ct_channels=ct_large_channels,
            dose_channels=dose_large_channels,
            use_ct=large_branch_use_ct,
        )

        # Combined embedding dimension
        branch1_dim  = dla_channels[-1] * 2          # CT + PET (no dose in branch1)
        branch2_dim  = self.branch2.out_dim
        combined_dim = branch1_dim + branch2_dim

        # Top-level fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, dual_scale_fusion_hidden),
            nn.BatchNorm1d(dual_scale_fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dual_scale_fusion_hidden, dual_scale_fusion_hidden // 2),
            nn.BatchNorm1d(dual_scale_fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dual_scale_fusion_hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        ct_small:   torch.Tensor,
        pet_small:  torch.Tensor,
        ct_large:   torch.Tensor,
        dose_large: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-instance (use_mil=False):
            ct_small, pet_small : (B, 1, D, H, W)
        MIL mode (use_mil=True):
            ct_small, pet_small : (B, N, 1, D, H, W)
        ct_large, dose_large    : (B, 1, D_l, H_l, W_l)  always single-instance
        returns logits          : (B,)
        """
        emb1  = self.branch1.encode(ct_small, pet_small)      # (B, branch1_dim)
        emb2  = self.branch2(ct_large, dose_large)             # (B, branch2_dim)
        fused = torch.cat([emb1, emb2], dim=1)                 # (B, combined_dim)
        return self.classifier(fused).squeeze(1)               # (B,)


# ── Global CT+Dose branch ─────────────────────────────────────────────────────

class DoseModulation(nn.Module):
    """
    Soft multiplicative dose modulation:
        x_modulated = ct * (1 + alpha * dose)

    alpha is a learnable scalar nn.Parameter, initialized to a small positive
    value so the modulation starts mild and the network can widen or narrow it.

    Dose is assumed to be normalised to roughly [0, 1] or [0, 2]; either way
    the initial modulation amplitude is at most alpha * dose_max ≈ 0.25–0.5,
    keeping activations stable at the start of training.
    """

    def __init__(self, alpha_init: float = 0.25):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, ct: torch.Tensor, dose: torch.Tensor) -> torch.Tensor:
        # ct:   (B, 1, D, H, W)  — CT normalised to [0, 1]
        # dose: (B, 1, D, H, W)  — dose normalised to ~[0, 1]
        mod_map = 1.0 + self.alpha * dose    # (B, 1, D, H, W)
        return ct * mod_map                  # (B, 1, D, H, W)


class GlobalCTDoseEncoder(nn.Module):
    """
    Lightweight 3-D CNN for full-volume dose-modulated CT.

    Four conv stages with stride (1,2,2) in XY (preserves the thin Z axis
    of typical CT), followed by AdaptiveAvgPool3d(1) which accepts any
    spatial input size.  A final linear projection maps to out_dim.

    Args:
        in_channels: 1  (dose-modulated CT is single-channel)
        channels:    output channels per stage, e.g. [16, 32, 64, 128]
        out_dim:     global token dimension
    """

    def __init__(
        self,
        in_channels: int       = 1,
        channels:    List[int] = None,
        out_dim:     int       = 128,
    ):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128]
        self.out_dim = out_dim

        layers = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            stride = (1, 2, 2) if i > 0 else 1   # first stage: no stride
            layers += [
                nn.Conv3d(c_in, c_out, kernel_size=3,
                          stride=stride, padding=1, bias=False),
                nn.BatchNorm3d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out

        self.encoder = nn.Sequential(*layers)
        self.gap     = nn.AdaptiveAvgPool3d(1)
        self.proj    = (nn.Linear(channels[-1], out_dim)
                        if channels[-1] != out_dim else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (B, 1, D, H, W)  — dose-modulated full CT
        feat = self.encoder(x)   # (B, channels[-1], D', H', W')
        feat = self.gap(feat)    # (B, channels[-1], 1, 1, 1)
        feat = feat.flatten(1)   # (B, channels[-1])
        return self.proj(feat)   # (B, out_dim)


class GatedFusion(nn.Module):
    """
    Sample-wise scalar gated fusion of ROI token and global token.

    Steps:
        z_global_proj = Linear(z_global)                      # (B, roi_dim)
        g             = sigmoid(Linear(ReLU(Linear(cat(...)))) # (B, 1)
        z_fused       = g * z_roi + (1 - g) * z_global_proj  # (B, roi_dim)

    g → 1: rely on ROI evidence;  g → 0: rely on global evidence.

    Args:
        roi_dim:     dimension of z_roi  (= 2 * dla_channels[-1])
        global_dim:  dimension of z_global (out_dim of GlobalCTDoseEncoder)
        gate_hidden: hidden dim of the 2-layer gate MLP
    """

    def __init__(self, roi_dim: int, global_dim: int, gate_hidden: int = 64):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, roi_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(roi_dim * 2, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, z_roi: torch.Tensor, z_global: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # z_roi:    (B, roi_dim)
        # z_global: (B, global_dim)
        z_g     = self.global_proj(z_global)                        # (B, roi_dim)
        g       = self.gate_mlp(torch.cat([z_roi, z_g], dim=1))    # (B, 1)
        z_fused = g * z_roi + (1.0 - g) * z_g                      # (B, roi_dim)
        return z_fused, g                                           # (B, roi_dim), (B, 1)


class CombinedPETCTDoseMILModel(nn.Module):
    """
    ROI PET/CT DLA + MIL branch  ⊕  full-volume CT+Dose global branch,
    fused via a sample-wise scalar gate.

    ROI branch  (identical to DualBranch3DCNN internals, use_dose=False)
    ──────────
      ct_roi  (B, N, 1, Dr, Hr, Wr) ──► DLA3D ──► GAP ──► emb_ct  (B*N, C)
      pet_roi (B, N, 1, Dr, Hr, Wr) ──► DLA3D ──► GAP ──► emb_pet (B*N, C)
                                                    concat (B*N, 2C)
                                                    view   (B, N, 2C)
                                       GatedAttentionMIL ──► z_roi (B, 2C)

    Global branch  (new)
    ─────────────
      DoseModulation: x_mod = ct_global * (1 + alpha * dose_global)
      GlobalCTDoseEncoder: x_mod ──► z_global (B, global_out_dim)

    Gated fusion  (new)
    ────────────
      g       = sigmoid(MLP([z_roi, proj(z_global)]))   (B, 1)
      z_fused = g * z_roi + (1-g) * proj(z_global)      (B, roi_dim)

    Classifier
    ──────────
      z_fused ──► Linear → Norm → ReLU → Dropout  ×2  ──► logit (B,)

    Args:
        dla_channels:         DLA stage channels for CT and PET backbones.
        dropout:              dropout probability in classifier.
        use_mil:              enable Gated-Attention MIL pooling.
        mil_attn_dim:         hidden dim of gated attention.
        use_missing_gate:     enable missing-PET gate (same as DualBranch3DCNN).
        missing_gate_hidden:  hidden dim of the PET-presence gate MLP.
        global_channels:      channel list for GlobalCTDoseEncoder.
        global_out_dim:       output dimension of GlobalCTDoseEncoder.
        dose_alpha_init:      initial value of the learnable alpha.
        gate_hidden:          hidden dim of the gated-fusion MLP.
        fusion_hidden:        hidden dim of the final classifier MLP.
    """

    def __init__(
        self,
        dla_channels:         List[int] = None,
        dropout:              float     = 0.5,
        use_mil:              bool      = True,
        mil_attn_dim:         int       = 128,
        use_missing_gate:     bool      = False,
        missing_gate_hidden:  int       = 64,
        global_channels:      List[int] = None,
        global_out_dim:       int       = 128,
        dose_alpha_init:      float     = 0.25,
        gate_hidden:          int       = 64,
        fusion_hidden:        int       = 256,
        num_classes:          int       = 1,
        use_unipair:          bool      = False,
    ):
        super().__init__()
        if dla_channels is None:
            dla_channels = [16, 32, 64, 128, 256]
        if global_channels is None:
            global_channels = [16, 32, 64, 128]

        self.use_mil          = use_mil
        self.use_missing_gate = use_missing_gate
        self.use_unipair      = use_unipair
        embed_c = dla_channels[-1]
        self.embed_c = embed_c             # stored for head_ct input slicing
        roi_dim = embed_c * 2               # CT + PET concat

        # ── ROI branch: reuse DLA3D + GatedAttentionMIL blocks ────────────────
        self.ct_branch  = DLA3D(in_channels=1, channels=dla_channels)
        self.pet_branch = DLA3D(in_channels=1, channels=dla_channels)

        if use_missing_gate:
            self.pet_gate = nn.Sequential(
                nn.Linear(embed_c + 1, missing_gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(missing_gate_hidden, 1),
                nn.Sigmoid(),
            )

        if use_mil:
            self.mil_attention = GatedAttentionMIL(roi_dim, hidden_dim=mil_attn_dim)

        # ── Global branch ─────────────────────────────────────────────────────
        self.dose_mod       = DoseModulation(alpha_init=dose_alpha_init)
        self.global_encoder = GlobalCTDoseEncoder(
            in_channels=1,
            channels=global_channels,
            out_dim=global_out_dim,
        )

        # ── Gated fusion ──────────────────────────────────────────────────────
        self.gate_fusion = GatedFusion(
            roi_dim=roi_dim,
            global_dim=global_out_dim,
            gate_hidden=gate_hidden,
        )

        # ── Classifier head(s) ────────────────────────────────────────────────
        def _make_head_g(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, fusion_hidden),
                nn.LayerNorm(fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                nn.LayerNorm(fusion_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden // 2, num_classes),
            )

        if use_missing_gate:
            # head_fused: paired patients (z_fused from gate_fusion, dim=roi_dim)
            # head_ct:    CT-only patients (z_roi CT slice, dim=embed_c)
            self.head_fused = _make_head_g(roi_dim)
            self.head_ct    = _make_head_g(embed_c)
        elif use_unipair:
            # Unipair mode: separate symmetric head for each modality.
            # head_ct  receives z_roi[:, :embed_c]        (CT embedding)
            # head_pet receives z_roi[:, embed_c:embed_c*2] (PET embedding)
            # Both heads run on ALL samples so DDP parameters stay symmetric.
            self.head_ct = nn.Sequential(
                nn.LayerNorm(embed_c),
                nn.Linear(embed_c, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden, num_classes),
            )
            self.head_pet = nn.Sequential(
                nn.LayerNorm(embed_c),
                nn.Linear(embed_c, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(roi_dim, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                nn.BatchNorm1d(fusion_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden // 2, num_classes),
            )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # dose_mod.alpha is already set by DoseModulation.__init__

    # ── ROI encoding (mirrors DualBranch3DCNN._encode with missing-gate fix) ──

    def _encode_roi(
        self,
        ct:          torch.Tensor,          # (B, 1, Dr, Hr, Wr)
        pet:         torch.Tensor,          # (B, 1, Dr, Hr, Wr)
        pet_present: torch.Tensor = None,   # (B,) float 0/1; None → all paired
    ) -> torch.Tensor:
        """Return fused ROI embedding (B, roi_dim = 2*C) for one set of instances."""
        emb_ct = F.adaptive_avg_pool3d(self.ct_branch(ct), 1).flatten(1)  # (B, C)

        if self.use_missing_gate:
            if pet_present is not None:
                pp     = pet_present.view(-1, 1).float()               # (B, 1)
                paired = pp.squeeze(1).bool()                           # (B,)
            else:
                pp     = emb_ct.new_ones(emb_ct.shape[0], 1)
                paired = torch.ones(emb_ct.shape[0], dtype=torch.bool,
                                    device=emb_ct.device)

            # Always run pet_branch on ALL inputs (DDP requires symmetric parameter
            # participation on every rank).  Hard mask (* pp) zeroes CT-only samples.
            emb_pet = F.adaptive_avg_pool3d(self.pet_branch(pet), 1).flatten(1)
            g       = self.pet_gate(torch.cat([emb_ct, pp], dim=1))    # (B, 1)
            emb_pet = g * emb_pet * pp                                  # hard mask
        else:
            emb_pet = F.adaptive_avg_pool3d(self.pet_branch(pet), 1).flatten(1)

        return torch.cat([emb_ct, emb_pet], dim=1)   # (B, roi_dim)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        ct_roi:           torch.Tensor,          # (B, N, 1, Dr, Hr, Wr)  MIL
                                                 # or (B, 1, Dr, Hr, Wr)  single
        pet_roi:          torch.Tensor,          # same shape as ct_roi
        ct_global:        torch.Tensor,          # (B, 1, D, H, W)  full CT
        dose_global:      torch.Tensor,          # (B, 1, D, H, W)  full dose
        pet_present:      torch.Tensor = None,   # (B,) for missing gate
        return_attn:      bool         = False,  # return MIL attention weights
        return_details:   bool         = False,  # return full analysis dict
        return_ct_logits: bool         = False,  # return aux CT-head logits on all patients
        modality:         torch.Tensor = None,   # (B,) float 0=CT / 1=PET; use_unipair only
    ):
        """
        Default:                                  returns logits (B,)
        return_attn=True:                         returns (logits, A)
        return_details=True:                      returns dict with all intermediate tokens
        return_ct_logits=True:                    returns (logits, logits_ct)  [mutually
                                                  exclusive with return_details]
        return_attn=True, return_ct_logits=True:  returns (logits, A, logits_ct)

        logits_ct is head_ct applied to the pure-CT ROI embedding on ALL patients,
        providing an auxiliary loss signal. Only meaningful when use_missing_gate=True;
        returns None for logits_ct otherwise.
        """
        # ── ROI branch ────────────────────────────────────────────────────────
        if self.use_mil:
            B, N    = ct_roi.shape[:2]
            ct_in   = ct_roi.view(B * N, *ct_roi.shape[2:])    # (B*N, 1, Dr, Hr, Wr)
            pet_in  = pet_roi.view(B * N, *pet_roi.shape[2:])  # (B*N, 1, Dr, Hr, Wr)
            pp_in   = (pet_present.repeat_interleave(N)
                       if pet_present is not None else None)    # (B*N,)
            emb     = self._encode_roi(ct_in, pet_in, pp_in)   # (B*N, roi_dim)
            emb     = emb.view(B, N, -1)                        # (B, N, roi_dim)
            z_roi, A = self.mil_attention(emb)                  # (B, roi_dim), (B, N, 1)
        else:
            z_roi = self._encode_roi(ct_roi, pet_roi, pet_present)  # (B, roi_dim)
            A     = None

        # ── Global branch ─────────────────────────────────────────────────────
        x_mod    = self.dose_mod(ct_global, dose_global)   # (B, 1, D, H, W)
        z_global = self.global_encoder(x_mod)              # (B, global_out_dim)

        # ── Gated fusion ──────────────────────────────────────────────────────
        z_fused, g = self.gate_fusion(z_roi, z_global)     # (B, roi_dim), (B, 1)

        # ── Classifier ────────────────────────────────────────────────────────
        if self.use_missing_gate:
            # head_fused: z_fused (full CT+PET+global representation)
            # head_ct:    z_roi CT slice (pure CT after MIL pooling, no PET, no global)
            emb_ct_only  = z_roi[:, :self.embed_c]                    # (B, embed_c)
            logits_fused = self.head_fused(z_fused).squeeze(1)        # (B,)
            logits_ct    = self.head_ct(emb_ct_only).squeeze(1)       # (B,)
            if pet_present is not None:
                pp = pet_present.bool()
                logits = torch.where(pp, logits_fused, logits_ct)
            else:
                logits = logits_fused
        elif self.use_unipair and modality is not None:
            # Unipair mode: both heads run on all samples for DDP symmetry
            emb_ct_only  = z_roi[:, :self.embed_c]                    # (B, embed_c)
            emb_pet_only = z_roi[:, self.embed_c:self.embed_c * 2]    # (B, embed_c)
            logits_ct_head  = self.head_ct(emb_ct_only).squeeze(1)    # (B,)
            logits_pet_head = self.head_pet(emb_pet_only).squeeze(1)  # (B,)
            mod = modality.bool()   # True = PET, False = CT
            logits    = torch.where(mod, logits_pet_head, logits_ct_head)
            logits_ct = None
        else:
            logits    = self.classifier(z_fused).squeeze(1)           # (B,)
            logits_ct = None

        if return_ct_logits:
            # return_ct_logits is mutually exclusive with return_details
            logits_ct_out = logits_ct if self.use_missing_gate else None
            if return_attn:
                return logits, A, logits_ct_out
            return logits, logits_ct_out

        if return_attn:
            return logits, A

        if return_details:
            return {
                "logits":       logits,
                "roi_token":    z_roi,
                "global_token": z_global,
                "fused_token":  z_fused,
                "gate_value":   g,
                "alpha":        self.dose_mod.alpha,
            }

        return logits


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from config import PATCH_SIZE, DLA_CHANNELS, EMBED_DIM, FUSION_HIDDEN, DROPOUT

    B = 2
    D, H, W = PATCH_SIZE                     # e.g. (24, 72, 72)
    ct  = torch.randn(B, 1, D, H, W)
    pet = torch.randn(B, 1, D, H, W)

    model = DualBranch3DCNN(
        dla_channels=DLA_CHANNELS,
        embed_dim=EMBED_DIM,
        fusion_hidden=FUSION_HIDDEN,
        dropout=DROPOUT,
    )
    model.eval()
    with torch.no_grad():
        out = model(ct, pet)
    print(f"Input:  CT {tuple(ct.shape)},  PET {tuple(pet.shape)}")
    print(f"Output: {tuple(out.shape)}")  # (B,)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
