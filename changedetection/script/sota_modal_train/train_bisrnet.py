import sys
sys.path.append('/mnt/user/zhang_haotian/FoBa')

import argparse
import os
import time

import numpy as np

from changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset, make_data_loader
from changedetection.utils_func.metrics import Evaluator
from changedetection.models.bisrnet import BiSRNet
from changedetection.BiSRNet_help.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity
from torch.optim.lr_scheduler import StepLR
from changedetection.utils_func.mcd_utils import accuracy, SCDD_eval_all, AverageMeter
torch.manual_seed(2025)            # 为CPU设置随机种子 2023
torch.cuda.manual_seed(2025)      # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(2025)  # 为所有GPU设置随机种子

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)


        self.deep_model = BiSRNet(
            in_channels=3,
            num_classes=7
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

        self.optim = optim.SGD(filter(lambda p: p.requires_grad, self.deep_model.parameters()),
                               lr=args.learning_rate, weight_decay=args.weight_decay,
                               momentum=args.momentum, nesterov=True)

        self.scheduler = StepLR(self.optim, step_size=10000, gamma=0.5)

    def training(self):
        best_kc = 0.0
        best_round = []
        criterion_sc = ChangeSimilarity().cuda()
        criterion = CrossEntropyLoss2d().cuda()
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, label_cd, label_clf_t1, label_clf_t2, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            # label_cd = label_cd.cuda().long()
            label_cd = label_cd.cuda().float()
            label_clf_t1 = label_clf_t1.cuda().long()
            label_clf_t2 = label_clf_t2.cuda().long()

            output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)

            self.optim.zero_grad()

            loss_seg = criterion(output_semantic_t1, label_clf_t1) * 0.5 + criterion(output_semantic_t2, label_clf_t2) * 0.5
            loss_bn = weighted_BCE_logits(output_1, label_cd)
            loss_sc = criterion_sc(output_semantic_t1[:, 1:], output_semantic_t2[:, 1:], label_cd)

            final_loss = loss_seg + loss_bn + loss_sc

            final_loss.backward()

            self.optim.step()
            self.scheduler.step()

            if (itera + 1) % 10 == 0:
                print(
                    f'iter is {itera + 1}, final_loss is {final_loss}, loss_seg is {loss_seg}, loss_bn is {loss_bn}, loss_sc is {loss_sc}')
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


                output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)

                labels_cd = labels_cd.cpu().numpy()
                labels_A = labels_clf_t1.cpu().numpy()
                labels_B = labels_clf_t2.cpu().numpy()

                change_mask = F.sigmoid(output_1).cpu().detach() > 0.5
                # change_mask = torch.argmax(output_1, axis=1).cpu().numpy()

                preds_A = torch.argmax(output_semantic_t1, dim=1).cpu().numpy()
                preds_B = torch.argmax(output_semantic_t2, dim=1).cpu().numpy()
                # import pdb;pdb.set_trace()
                preds_scd = (preds_A - 1) * 6 + preds_B
                preds_scd[change_mask.squeeze(1).long() == 0] = 0

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
    parser.add_argument('--cfg', type=str,
                        default=r'/mnt/user/zhang_haotian/FoBa/changedetection/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--dataset', type=str, default='SECOND')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default=r'/mnt/dataset/zhanghaotian/CDdataset/SECOND/train')
    parser.add_argument('--train_data_list_path', type=str, default=r'/mnt/dataset/zhanghaotian/CDdataset/SECOND/list/train_list.txt')
    parser.add_argument('--test_dataset_path', type=str, default=r'/mnt/dataset/zhanghaotian/CDdataset/SECOND/test')
    parser.add_argument('--test_data_list_path', type=str, default=r'/mnt/dataset/zhanghaotian/CDdataset/SECOND/list/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=800000)
    parser.add_argument('--model_type', type=str, default='BiSRNet')
    parser.add_argument('--model_param_path', type=str, default='/mnt/dataset/zhanghaotian/ChangeSCD/exp/saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
