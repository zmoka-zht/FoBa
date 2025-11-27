import sys

sys.path.append('/mnt/user/zhang_haotian/FoBa')

import argparse
import os
import time

import numpy as np

from changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset, make_data_loader
from changedetection.utils_func.metrics import Evaluator
from changedetection.models.FoBaSCD import FoBaMambaBased, FoBaTransformerBased

from changedetection.utils_func.mcd_utils import accuracy, SCDD_eval_all, AverageMeter
from tqdm import tqdm
import imageio
import numpy as np
import os
import matplotlib.pyplot as plt


ori_label_value_dict = {
        "background": (255, 255, 255),
        "Sparse woodland": (97, 101, 63),
        "Low vegetation": (238, 238, 217),
        "Woodland": (197, 196, 123),
        "Playground": (214, 203, 201),
        "Low building": (98, 180, 252),
        "General building": (194, 206, 218),
        "Unpaved Road": (139, 115, 227),
        "Bare Land": (206, 232, 255),
        "Construction land": (115, 83, 73),
        "Parking Lot": (202, 198, 215),
        "Others": (209, 172, 139),
        "River": (250, 228, 220),
        "Impervious surfaces": (210, 209, 197),
        "Paved Road": (72, 57, 113),
        "High building": (52, 91, 121),
        "Water": (234, 165, 140)
        }

target_label_value_dict = {
        "background": 0,
        "Sparse woodland": 1,
        "Low vegetation": 2,
        "Woodland": 3,
        "Playground": 4,
        "Low building": 5,
        "General building": 6,
        "Unpaved Road": 7,
        "Bare Land": 8,
        "Construction land": 9,
        "Parking Lot": 10,
        "Others": 11,
        "River": 12,
        "Impervious surfaces": 13,
        "Paved Road": 14,
        "High building": 15,
        "Water": 16
        }


def map_labels_to_colors(labels, ori_label_value_dict, target_label_value_dict):
    # Reverse the target_label_value_dict to get a mapping from target labels to original labels
    target_to_ori = {v: k for k, v in target_label_value_dict.items()}

    # Initialize an empty 3D array for the color-mapped labels
    H, W = labels.shape
    color_mapped_labels = np.zeros((H, W, 3), dtype=np.uint8)

    for target_label, ori_label in target_to_ori.items():
        # Find where the label matches the current target label
        mask = labels == target_label

        # Map these locations to the corresponding color value
        color_mapped_labels[mask] = ori_label_value_dict[ori_label]

    return color_mapped_labels


class Inference(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.deep_model = FoBaMambaBased(
        #self.deep_model = FoBaTransformerBased(
            output_cd=2,
            output_clf=17,
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        self.deep_model = self.deep_model.cuda()
        
        self.change_map_T1_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'change_map_T1')
        self.change_map_T2_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'change_map_T2')
        if not os.path.exists(self.change_map_T1_saved_path):
            os.makedirs(self.change_map_T1_saved_path)
        if not os.path.exists(self.change_map_T2_saved_path):
            os.makedirs(self.change_map_T2_saved_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)
            print('pretrained weight has been loaded')

        self.deep_model.eval()

    def infer(self):
        torch.cuda.empty_cache()
        dataset = SemanticChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 512, None,
                                                'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        acc_meter = AverageMeter()
        torch.cuda.empty_cache()
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_cd, labels_clf_t1, labels_clf_t2, names  = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_cd = labels_cd.cuda().long()
                labels_clf_t1 = labels_clf_t1.cuda().long()
                labels_clf_t2 = labels_clf_t2.cuda().long()

                output_1, output_semantic_t1, output_semantic_t2, pred_mask_2, pred_mask_3, pred_mask_4 = self.deep_model(
                    pre_change_imgs, post_change_imgs)

                labels_cd = labels_cd.cpu().numpy()
                labels_A = labels_clf_t1.cpu().numpy()
                labels_B = labels_clf_t2.cpu().numpy()

                change_mask = torch.argmax(output_1, axis=1).cpu().numpy()

                preds_A = torch.argmax(output_semantic_t1, dim=1).cpu().numpy()
                preds_B = torch.argmax(output_semantic_t2, dim=1).cpu().numpy()

                preds_scd = (preds_A - 1) * 16 + preds_B
                preds_scd[change_mask == 0] = 0

                labels_scd = (labels_A - 1) * 16 + labels_B
                labels_scd[labels_cd == 0] = 0

                for (pred_scd, label_scd) in zip(preds_scd, labels_scd):
                    acc_A, valid_sum_A = accuracy(pred_scd, label_scd)
                    preds_all.append(pred_scd)
                    labels_all.append(label_scd)
                    acc = acc_A
                    acc_meter.update(acc)

                #preds_A = (preds_A * change_mask.squeeze())
                #preds_B = (preds_B * change_mask.squeeze())

                #change_map_T1 = map_labels_to_colors(np.squeeze(preds_A), ori_label_value_dict=ori_label_value_dict, target_label_value_dict=target_label_value_dict)
                #change_map_T2 = map_labels_to_colors(np.squeeze(preds_B), ori_label_value_dict=ori_label_value_dict, target_label_value_dict=target_label_value_dict)
                #image_name = names[0][0:-4] + f'.png'

                #imageio.imwrite(os.path.join(self.change_map_T1_saved_path, image_name), change_map_T1.astype(np.uint8))
                #imageio.imwrite(os.path.join(self.change_map_T2_saved_path, image_name), change_map_T2.astype(np.uint8))


        kappa_n0, Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, 257)
        print(f'Kappa coefficient rate is {kappa_n0}, F1 is {Fscd}, OA is {acc_meter.avg}, '
              f'mIoU is {IoU_mean}, SeK is {Sek}')


        print(f'Inference stage is done!')


def main():
    parser = argparse.ArgumentParser(description="Inference on LEVIRSCD dataset")
    parser.add_argument('--cfg', type=str,
                        default=r'/mnt/user/zhang_haotian/FoBa/changedetection/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str,
                        default=r"/mnt/dataset/zhanghaotian/PretrainWeight/vision_mamba/vssm_base_0229_ckpt_epoch_237.pth")

    parser.add_argument('--dataset', type=str, default='LEVIRSCD')
    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--test_dataset_path', type=str,
                        default=r'/mnt/user/zhang_haotian/PycharmProject/LevirSCD/test')
    parser.add_argument('--test_data_list_path', type=str,
                        default=r'/mnt/user/zhang_haotian/PycharmProject/LevirSCD/list/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--model_type', type=str, default='FoBaTransformerBased')
    parser.add_argument('--result_saved_path', type=str, default='/mnt/dataset/zhanghaotian/ChangeSCD/exp/saved_models/SECOND/FoBa_1764142282.9340718/Inference_result')

    parser.add_argument('--resume', type=str, default=r"/mnt/user/zhang_haotian/FoBa/levirscd_mamba.pth")

    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    infer = Inference(args)
    infer.infer()


if __name__ == "__main__":
    main()
