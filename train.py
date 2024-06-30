import torch
import torch.nn as nn
import os
from networks import get_model
from datasets import data_merge
from optimizers import get_optimizer
from torch.utils.data import DataLoader
from transformers import *
from utils import *
from configs import parse_args
import time
import numpy as np
import random
import sys
from loss import *

torch.manual_seed(1226)
np.random.seed(1226)
random.seed(1226)


def main(args):
    if args.data_dir == "":
        print("Please provide the data directory path through data_dir arg.")
        sys.exit()
    if args.loo_domain == "":
        print("Please provide the leave_one_out domain name through loo_domain arg.")
        sys.exit()
    
    data_bank = data_merge(args.data_dir)
    # define train loader
    if args.trans in ["o"]:
        train_set = data_bank.get_datasets(train=True, loo_domain=args.loo_domain, img_size=args.img_size, transform=transformer_train())
    elif args.trans in ["p"]:
        train_set = data_bank.get_datasets(train=True, loo_domain=args.loo_domain, img_size=args.img_size, transform=transformer_train_pure())
    elif args.trans in ["I"]:
        train_set = data_bank.get_datasets(train=True, loo_domain=args.loo_domain, img_size=args.img_size, transform=transformer_train_ImageNet())
    else:
        raise Exception
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    max_iter = args.num_epochs*len(train_loader)
    # define model
    model = get_model().cuda()
    # def optimizer
    optimizer = get_optimizer(
        args.optimizer, model, 
        lr=args.base_lr,
        momentum=args.momentum, 
        weight_decay=args.weight_decay)
    # def scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-5)
    model = nn.DataParallel(model).cuda()

    start_epoch = 0

    result_name = "loo_" + args.loo_domain
    # make dirs
    model_root_path = os.path.join(args.result_path, result_name, "model")
    check_folder(model_root_path)
    score_root_path = os.path.join(args.result_path, result_name, "score")
    check_folder(score_root_path)


    # define loss
    binary_fuc = nn.CrossEntropyLoss()
    contra_func = ContrastLoss()
    supcon_func = SupContrast()

    # metrics
    eva = {
        "best_epoch": -1,
        "best_HTER": 100,
        "best_auc": -100
    }

    if (args.checkpoint_path != ""):
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        eva = checkpoint['best']
        args = checkpoint['args']

    for epoch in range(start_epoch, args.num_epochs):
        binary_loss_record = AverageMeter()
        constra_loss_record = AverageMeter()
        supcon_loss_record = AverageMeter()
        loss_record = AverageMeter()
        # train
        model.train()
        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()
            image_x, label, UUID = sample_batched["image_x"].cuda(), sample_batched["label"].cuda(), sample_batched["UUID"].cuda()
            # train process
            rand_idx = torch.randperm(image_x.shape[0])
            fea_x1, cls_x1_x1, fea_x1_x1, fea_x1_x2 = model(image_x, image_x[rand_idx, :, :, :])

            binary_loss = binary_fuc(cls_x1_x1, label[:, 0].long())

            contrast_label = label[:, 0].long() == label[rand_idx, 0].long()
            contrast_label = torch.where(contrast_label==True, 1, -1)
            constra_loss = contra_func(fea_x1_x1, fea_x1_x2, contrast_label) * args.lambda_contrast

            fea_x1_ensemble = torch.cat([fea_x1.unsqueeze(1), fea_x1.unsqueeze(1)], dim=1)
            supcon_loss = supcon_func(fea_x1_ensemble, UUID.long() * 10 + label[:, 0].long()) * args.lambda_supcon
            loss_all = binary_loss + constra_loss + supcon_loss

            n = image_x.shape[0]
            binary_loss_record.update(binary_loss.data, n)
            constra_loss_record.update(constra_loss.data, n)
            supcon_loss_record.update(supcon_loss.data, n)
            loss_record.update(loss_all.data, n)

            model.zero_grad()
            loss_all.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            if i % args.print_freq == args.print_freq - 1:
                print("Epoch:{:d}, mini-batch:{:d}, lr={:.5f}, binary_loss={:.4f}, constra_loss={:.4f}, supcon_loss={:.4f}, Loss={:.4f}".format(epoch + 1, i + 1, lr, binary_loss_record.avg, constra_loss_record.avg, supcon_loss_record.avg, loss_record.avg))
            scheduler.step()

        # whole epoch average
        print("Epoch:{:d}, Train: lr={:.5f}, Loss={:.4f}".format(epoch + 1, lr, loss_record.avg))

        # test
        epoch_test = 1
        if epoch % epoch_test == epoch_test-1:
            if args.trans in ["o", "p"]:
                test_data_dic = data_bank.get_datasets(train=False, loo_domain=args.loo_domain, img_size=args.img_size, transform=transformer_test_video())
            elif args.trans in ["I"]:
                test_data_dic = data_bank.get_datasets(train=False, loo_domain=args.loo_domain, img_size=args.img_size, transform=transformer_test_video_ImageNet())
            else:
                raise Exception
            score_path = os.path.join(score_root_path, "Epoch_{}".format(epoch+1))
            check_folder(score_path)
            for i, test_name in enumerate(test_data_dic.keys()):
                print("[{}/{}]Validating {}...".format(i+1, len(test_data_dic), test_name))
                test_set = test_data_dic[test_name]
                test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                HTER, auc_test = validate(model, test_loader, score_path, epoch, name=test_name)
                if auc_test-HTER>=eva["best_auc"]-eva["best_HTER"]:
                    eva["best_auc"] = auc_test
                    eva["best_HTER"] = HTER
                    eva["best_epoch"] = epoch+1
                    model_path = os.path.join(model_root_path, "{}_best.pth".format(result_name))
                    torch.save({
                        'epoch': epoch+1,
                        'state_dict':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),
                        'best':{"best_epoch": eva["best_epoch"], "best_HTER": eva["best_HTER"], "best_auc": eva["best_auc"]},
                        'args':args,
                    }, model_path)
                    print("Model saved to {}".format(model_path))
                print("[Best result] Epoch:{}, HTER={:.4f}, AUC={:.4f}".format(eva["best_epoch"],  eva["best_HTER"], eva["best_auc"]))
            model_path = os.path.join(model_root_path, "{}_recent.pth".format(result_name))
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'best':{"best_epoch": eva["best_epoch"], "best_HTER": eva["best_HTER"], "best_auc": eva["best_auc"]},
                'args':args,
            }, model_path)
            print("Model saved to {}".format(model_path))


def validate(model, test_loader, score_root_path, epoch, name=""):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        scores_list = []
        for i, sample_batched in enumerate(test_loader):
            image_x, label = sample_batched["image_x"].cuda(), sample_batched["label"].cuda()
            fea_x1, cls_x1_x1, fea_x1_x1, fea_x1_x2 = model(image_x, image_x)
            score_norm = torch.softmax(cls_x1_x1, dim=1)[:, 1]
            for ii in range(image_x.shape[0]):
                scores_list.append("{} {}\n".format(score_norm[ii], label[ii][0]))
            
        map_score_val_filename = os.path.join(score_root_path, "{}_score.txt".format(name))
        print("score: write val scores to {}".format(map_score_val_filename))
        with open(map_score_val_filename, 'w') as file:
            file.writelines(scores_list)

        test_ACC, FPR, FRR, HTER, auc_test, test_err = performances_val(map_score_val_filename)
        print("## {} score:".format(name))
        print("Epoch:{:d}, val_result:  val_ACC={:.4f}, FPR={:.4f}, FRR={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}".format(epoch+1, test_ACC, FPR, FRR, HTER, auc_test, test_err))
        print("Validate phase cost {:.4f}s".format(time.time()-start_time))
    return HTER, auc_test

    
if __name__ == '__main__':
    args = parse_args()
    main(args=args)
