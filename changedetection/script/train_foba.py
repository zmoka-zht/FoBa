import sys

sys.path.append('/mnt/user/zhang_haotian/FoBa')

import argparse
import os
import time

import numpy as np
import torch.nn as nn
from changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset, make_data_loader
from changedetection.utils_func.metrics import Evaluator
from changedetection.models.FoBaSCD import FoBaMambaBased, FoBaTransformerBased

import changedetection.utils_func.lovasz_loss as L
from torch.optim.lr_scheduler import StepLR
from changedetection.utils_func.mcd_utils import accuracy, SCDD_eval_all, AverageMeter
torch.manual_seed(2025)
torch.cuda.manual_seed(2025)
torch.cuda.manual_seed_all(2025)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)


        self.deep_model = FoBaMambaBased(
        #self.deep_model = FoBaTransformerBased(
            output_cd=2,
            output_clf=7,
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
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

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

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

        self.scheduler = StepLR(self.optim, step_size=10000, gamma=0.5)

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, label_cd, label_clf_t1, label_clf_t2, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            label_cd = label_cd.cuda().long()
            label_clf_t1 = label_clf_t1.cuda().long()
            label_clf_t2 = label_clf_t2.cuda().long()

            output_1, output_semantic_t1, output_semantic_t2, pred_mask_2, pred_mask_3, pred_mask_4 = self.deep_model(
                pre_change_imgs, post_change_imgs)

            self.optim.zero_grad()

            ### mask guided supervised loss
            mask2_loss_cd = F.binary_cross_entropy_with_logits(pred_mask_2, label_cd.float().unsqueeze(1))
            mask3_loss_cd = F.binary_cross_entropy_with_logits(pred_mask_3, label_cd.float().unsqueeze(1))
            mask4_loss_cd = F.binary_cross_entropy_with_logits(pred_mask_4, label_cd.float().unsqueeze(1))

            ce_loss_cd = F.cross_entropy(output_1, label_cd)
            lovasz_loss_cd = L.lovasz_softmax(F.softmax(output_1, dim=1), label_cd)

            ce_loss_clf_t1 = F.cross_entropy(output_semantic_t1, label_clf_t1)
            lovasz_loss_clf_t1 = L.lovasz_softmax(F.softmax(output_semantic_t1, dim=1), label_clf_t1)

            ce_loss_clf_t2 = F.cross_entropy(output_semantic_t2, label_clf_t2)
            lovasz_loss_clf_t2 = L.lovasz_softmax(F.softmax(output_semantic_t2, dim=1), label_clf_t2)

            similarity_mask = (label_clf_t1 == 0).float().unsqueeze(1).expand_as(output_semantic_t1)

            # Similarity loss calculation (e.g., MSE)
            similarity_loss = F.mse_loss(F.softmax(output_semantic_t1, dim=1) * similarity_mask,
                                         F.softmax(output_semantic_t2, dim=1) * similarity_mask, reduction='mean')

            main_loss = ce_loss_cd + 0.5 * (ce_loss_clf_t1 + ce_loss_clf_t2 + 0.5 * similarity_loss) + 0.75 * (
                    lovasz_loss_cd + 0.5 * (lovasz_loss_clf_t1 + lovasz_loss_clf_t2)) + 0.5 * (
                                    mask2_loss_cd + mask3_loss_cd
                                    + mask4_loss_cd)
            final_loss = main_loss
            final_loss.backward()

            self.optim.step()
            self.scheduler.step()

            if (itera + 1) % 10 == 0:
                print(
                    f'iter is {itera + 1}, change detection loss is {ce_loss_cd + lovasz_loss_cd}, classification loss is {(ce_loss_clf_t1 + ce_loss_clf_t2 + lovasz_loss_clf_t1 + lovasz_loss_clf_t2) / 2}')
                if (itera + 1) >= 3000:
                    if (itera + 1) % 500 == 0:
                        self.deep_model.eval()
                        kappa_n0, Fscd, IoU_mean, Sek, oa = self.validation()
                        if Sek > best_kc:
                            torch.save(self.deep_model.state_dict(),
                                       os.path.join(self.model_save_path, f'{itera + 1}_model_{Sek}.pth'))
                            best_kc = Sek
                            best_round = [kappa_n0, Fscd, IoU_mean, Sek, oa]
                        self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        dataset = SemanticChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, self.args.crop_size, None,
                                                'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        acc_meter = AverageMeter()

        preds_all = []
        labels_all = []
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels_cd, labels_clf_t1, labels_clf_t2, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_cd = labels_cd.cuda().long()
                labels_clf_t1 = labels_clf_t1.cuda().long()
                labels_clf_t2 = labels_clf_t2.cuda().long()

                output_1, output_semantic_t1, output_semantic_t2, pred_mask_2, pred_mask_3, pred_mask_4 = self.deep_model(pre_change_imgs, post_change_imgs)

                labels_cd = labels_cd.cpu().numpy()
                labels_A = labels_clf_t1.cpu().numpy()
                labels_B = labels_clf_t2.cpu().numpy()

                change_mask = torch.argmax(output_1, axis=1).cpu().numpy()

                preds_A = torch.argmax(output_semantic_t1, dim=1).cpu().numpy()
                preds_B = torch.argmax(output_semantic_t2, dim=1).cpu().numpy()

                preds_scd = (preds_A - 1) * 6 + preds_B
                preds_scd[change_mask == 0] = 0

                labels_scd = (labels_A - 1) * 6 + labels_B
                labels_scd[labels_cd == 0] = 0

                for (pred_scd, label_scd) in zip(preds_scd, labels_scd):
                    acc_A, valid_sum_A = accuracy(pred_scd, label_scd)
                    preds_all.append(pred_scd)
                    labels_all.append(label_scd)
                    acc = acc_A
                    acc_meter.update(acc)

        kappa_n0, Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, 37)
        print(f'Kappa coefficient rate is {kappa_n0}, F1 is {Fscd}, OA is {acc_meter.avg}, '
              f'mIoU is {IoU_mean}, SeK is {Sek}')

        return kappa_n0, Fscd, IoU_mean, Sek, acc_meter.avg


def main():
    parser = argparse.ArgumentParser(description="Training on SECOND dataset")
    parser.add_argument('--cfg', type=str, #vssm_small_224.yaml vssm_base_224.yaml vssm_tiny_224_0229flex.yaml
                        default=r'/mnt/user/zhang_haotian/FoBa/changedetection/configs/vssm1/vssm_small_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str, #vssm_small_0229_ckpt_epoch_222.pth vssm_base_0229_ckpt_epoch_237.pth vssm_tiny_0230_ckpt_epoch_262.pth
                        default=r"/mnt/dataset/zhanghaotian/PretrainWeight/vision_mamba/vssm_small_0229_ckpt_epoch_222.pth")

    parser.add_argument('--dataset', type=str, default='SECOND')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default=r'/mnt/dataset/zhanghaotian/CDdataset/SECOND/train')
    parser.add_argument('--train_data_list_path', type=str,
                        default=r'/mnt/dataset/zhanghaotian/CDdataset/SECOND/list/train_list.txt')
    parser.add_argument('--test_dataset_path', type=str, default=r'/mnt/dataset/zhanghaotian/CDdataset/SECOND/test')
    parser.add_argument('--test_data_list_path', type=str,
                        default=r'/mnt/dataset/zhanghaotian/CDdataset/SECOND/list/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=480000)
    parser.add_argument('--model_type', type=str, default='FoBa')
    parser.add_argument('--model_param_path', type=str, default='/mnt/dataset/zhanghaotian/ChangeSCD/exp/saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
