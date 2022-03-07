import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Physio
from dataset_physio import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="physio_fold0_20220306_131630/")
parser.add_argument("--nsample", type=int, default=100)


parser.add_argument("--dataset", type=str, default='ecg')
parser.add_argument("--dataroot", type=str, default='/data/dataset/siagan')
parser.add_argument("--nc", type=int, default=1)
parser.add_argument("--folder", type=int, default=0)
parser.add_argument("--istest", type=int, default=0)
parser.add_argument("--n_aug", type=int, default=0)
parser.add_argument("--batchsize", type=int, default=32)
parser.add_argument("--workers", type=int, default=1)

args = parser.parse_args()
print(args)

#----------------------store setting in json file--------------------------
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/physio_fold" + str(args.nfold) + "_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


#----------------------Load data--------------------------
# train_loader, valid_loader, test_loader = get_dataloader(
#     seed=args.seed,
#     nfold=args.nfold,
#     batch_size=config["train"]["batch_size"],
#     missing_ratio=config["model"]["test_missing_ratio"],
# )
from data import load_data

loader=load_data(args)
train_loader, valid_loader, test_loader = loader['train'], loader['val'], loader['test']
#------------------------Model----------------------------
model = CSDI_Physio(config, args.device).to(args.device)

#-------------------Train & Evaluate----------------------
if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)