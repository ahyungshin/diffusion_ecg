import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle

import os
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score,classification_report,confusion_matrix, precision_recall_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

                if foldername != "":
                    torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []


        an_scores = torch.zeros(size=(len(test_loader.dataset),), dtype=torch.float32).cuda() # [27107]
        gt_labels = torch.zeros(size=(len(test_loader.dataset),), dtype=torch.long).cuda()

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it):
                output = model.evaluate(test_batch, nsample)
                label = test_batch[1]

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                # eval_points = eval_points.permute(0, 2, 1)
                # observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)[0] # pred: samples_median / target: c_target #[bs,320,1]

                error = torch.mean(
                    torch.pow((samples_median.view(samples_median.shape[0], -1) - c_target.view(c_target.shape[0], -1)), 2),
                    dim=1)
                    
                an_scores[batch_no*32 : batch_no*32+error.size(0)] = error.reshape(error.size(0))
                gt_labels[batch_no*32 : batch_no*32+error.size(0)] = label.reshape(error.size(0))

            # Scale error vector between [0, 1]
            # if scale:
            #     an_scores = (an_scores - torch.min(an_scores)) / (torch.max(an_scores) - torch.min(an_scores))

            y_=gt_labels.cpu().numpy()
            y_pred=an_scores.cpu().numpy()

        over_all = y_pred
        over_all_gt = y_


        min_score,max_score=np.min(over_all),np.max(over_all)

        rocprc,rocauc,prcauc,best_th,best_f1=evaluate2(over_all_gt,(over_all-min_score)/(max_score-min_score))
        print("#############################")
        print("########  Result  ###########")
        # print("ap:{}".format(aucprc))
        print("prc:{}".format(rocprc))
        print("auc:{}".format(rocauc))
        print("best th:{} --> best f1:{}".format(best_th,best_f1))

        with open(os.path.join(save_dir,"res-record.txt"),'w') as f:
            f.write("auc_prc:{}\n".format(aucprc))
            f.write("auc_roc:{}\n".format(aucroc))
            f.write("best th:{} --> best f1:{}".format(best_th, best_f1))



            #     all_target.append(c_target)
            #     all_evalpoint.append(eval_points)
            #     all_observed_point.append(observed_points)
            #     all_observed_time.append(observed_time)
            #     all_generated_samples.append(samples)

            #     mse_current = (
            #         ((samples_median.values - c_target) * eval_points) ** 2
            #     ) * (scaler ** 2)
            #     mae_current = (
            #         torch.abs((samples_median.values - c_target) * eval_points) 
            #     ) * scaler

            #     mse_total += mse_current.sum().item()
            #     mae_total += mae_current.sum().item()
            #     evalpoints_total += eval_points.sum().item()

            #     it.set_postfix(
            #         ordered_dict={
            #             "rmse_total": np.sqrt(mse_total / evalpoints_total),
            #             "mae_total": mae_total / evalpoints_total,
            #             "batch_no": batch_no,
            #         },
            #         refresh=True,
            #     )

            # with open(
            #     foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            # ) as f:
            #     all_target = torch.cat(all_target, dim=0)
            #     all_evalpoint = torch.cat(all_evalpoint, dim=0)
            #     all_observed_point = torch.cat(all_observed_point, dim=0)
            #     all_observed_time = torch.cat(all_observed_time, dim=0)
            #     all_generated_samples = torch.cat(all_generated_samples, dim=0)

            #     pickle.dump(
            #         [
            #             all_generated_samples,
            #             all_target,
            #             all_evalpoint,
            #             all_observed_point,
            #             all_observed_time,
            #             scaler,
            #             mean_scaler,
            #         ],
            #         f,
            #     )

            # CRPS = calc_quantile_CRPS(
            #     all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            # )

            # with open(
            #     foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            # ) as f:
            #     pickle.dump(
            #         [
            #             np.sqrt(mse_total / evalpoints_total),
            #             mae_total / evalpoints_total,
            #             CRPS,
            #         ],
            #         f,
            #     )
            #     print("RMSE:", np.sqrt(mse_total / evalpoints_total))
            #     print("MAE:", mae_total / evalpoints_total)
            #     print("CRPS:", CRPS)

def evaluate2(labels, scores,res_th=None, saveto=None):
    '''
    metric for auc/ap
    :param labels:
    :param scores:
    :param res_th:
    :param saveto:
    :return:
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # True/False Positive Rates.
    fpr, tpr, ths = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)


    # Use AUC function to calculate the area under the curve of precision recall curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    prc_auc = auc(recall, precision)



    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    # best f1
    best_f1 = 0
    best_threshold = 0
    for threshold in ths:
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores >= threshold] = 1
        tmp_scores[tmp_scores < threshold] = 0
        cur_f1 = f1_score(labels, tmp_scores)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = threshold
    #threshold f1
    if res_th is not None and saveto is  not None:
        tmp_scores = scores.copy()
        tmp_scores[tmp_scores >= res_th] = 1
        tmp_scores[tmp_scores < res_th] = 0
        print(classification_report(labels,tmp_scores))
        print(confusion_matrix(labels,tmp_scores))
    auc_prc=average_precision_score(labels,scores)
    return auc_prc,roc_auc,prc_auc,best_threshold,best_f1
