import sys
sys.path.append('/mnt/user/zhang_haotian/FoBa')

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from changedetection.models.Mamba_backbone import Backbone_VSSM
from classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from changedetection.models.FoBaDecoder import (ChangeDecoder, BaseChangeDecoder,
                                                FoBaDecoderSSM, FoBaDecoderTransformer)



from changedetection.models.SemanticDecoder import SemanticDecoder, BaseSemanticDecoder

class STMambaSCD(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained,  **kwargs):
        super(STMambaSCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)


        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )


        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)


    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output_bcd = self.decoder_bcd(pre_features, post_features)
        output_T1 = self.decoder_T1(pre_features)
        output_T2 = self.decoder_T2(post_features)


        output_bcd = self.main_clf_cd(output_bcd)
        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode='bilinear')

        output_T1 = self.aux_clf(output_T1)
        output_T1 = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')
        
        output_T2 = self.aux_clf(output_T2)
        output_T2 = F.interpolate(output_T2, size=post_data.size()[-2:], mode='bilinear')

        return output_bcd, output_T1, output_T2


class BaseMambaSCD(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained, **kwargs):
        super(BaseMambaSCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder_bcd = BaseChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T1 = BaseSemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.decoder_T2 = BaseSemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output_bcd = self.decoder_bcd(pre_features, post_features)
        output_T1 = self.decoder_T1(pre_features)
        output_T2 = self.decoder_T2(post_features)

        output_bcd = self.main_clf_cd(output_bcd)
        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode='bilinear')

        output_T1 = self.aux_clf(output_T1)
        output_T1 = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')

        output_T2 = self.aux_clf(output_T2)
        output_T2 = F.interpolate(output_T2, size=post_data.size()[-2:], mode='bilinear')

        return output_bcd, output_T1, output_T2


class FoBaMambaBased(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained, **kwargs):
        super(FoBaMambaBased, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder.channel_first

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = FoBaDecoderSSM(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)
        self.aux_clf2 = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        output_feature, pred_mask_2, pred_mask_3, pred_mask_4 = self.decoder(pre_features, post_features)
        output_bcd = self.main_clf_cd(output_feature)
        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode='bilinear')

        output_T1 = self.aux_clf(output_feature)
        output_T1 = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')

        output_T2 = self.aux_clf2(output_feature)
        output_T2 = F.interpolate(output_T2, size=post_data.size()[-2:], mode='bilinear')

        # f-bg mask supervision
        pred_mask_2 = F.interpolate(pred_mask_2, size=pre_data.size()[-2:], mode='bilinear')
        pred_mask_3 = F.interpolate(pred_mask_3, size=pre_data.size()[-2:], mode='bilinear')
        pred_mask_4 = F.interpolate(pred_mask_4, size=pre_data.size()[-2:], mode='bilinear')

        return output_bcd, output_T1, output_T2, pred_mask_2, pred_mask_3, pred_mask_4

class FoBaTransformerBased(nn.Module):
    def __init__(self, output_cd, output_clf, pretrained, **kwargs):
        super(FoBaTransformerBased, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        self.channel_first = self.encoder.channel_first

        print(self.channel_first)

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = FoBaDecoderTransformer(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf_cd = nn.Conv2d(in_channels=128, out_channels=output_cd, kernel_size=1)
        self.aux_clf = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)
        self.aux_clf2 = nn.Conv2d(in_channels=128, out_channels=output_clf, kernel_size=1)

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        output_feature, pred_mask_2, pred_mask_3, pred_mask_4 = self.decoder(pre_features, post_features)

        output_bcd = self.main_clf_cd(output_feature)
        output_bcd = F.interpolate(output_bcd, size=pre_data.size()[-2:], mode='bilinear')

        output_T1 = self.aux_clf(output_feature)
        output_T1 = F.interpolate(output_T1, size=pre_data.size()[-2:], mode='bilinear')

        output_T2 = self.aux_clf2(output_feature)
        output_T2 = F.interpolate(output_T2, size=post_data.size()[-2:], mode='bilinear')

        # gudied mask supervision
        pred_mask_2 = F.interpolate(pred_mask_2, size=pre_data.size()[-2:], mode='bilinear')
        pred_mask_3 = F.interpolate(pred_mask_3, size=pre_data.size()[-2:], mode='bilinear')
        pred_mask_4 = F.interpolate(pred_mask_4, size=pre_data.size()[-2:], mode='bilinear')

        return output_bcd, output_T1, output_T2, pred_mask_2, pred_mask_3, pred_mask_4



if __name__ == "__main__":
    image = torch.randn(1, 3, 512, 512).to("cuda:0")
    model = FoBaMambaBased(
            output_cd=2,
            output_clf=7,
            pretrained=r"/mnt/dataset/zhanghaotian/PretrainWeight/vision_mamba/vssm_tiny_0230_ckpt_epoch_262.pth",
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[ 2, 2, 4, 2 ],
            dims=96,
            # ===================
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v3noz",
            # ===================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # ===================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v3",
            patchembed_version="v2",
            gmlp=False,
            use_checkpoint=False,
        ).to("cuda:0")
    res = model(image, image)
    print(res[0].shape)
    print(res[1].shape)
    print(res[2].shape)
    print(res[3].shape)
    print(res[4].shape)
    print(res[5].shape)