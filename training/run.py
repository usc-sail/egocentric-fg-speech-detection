import argparse
import os
import ast
import pickle
import sys
import time
import torch
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import numpy as np
from traintest import train, validate
sys.path.append('../')
from models import vit_model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument('--dataset-mean', type=float)
parser.add_argument('--dataset-std', type=float)
parser.add_argument('--freqm', type=int)
parser.add_argument('--timem', type=int)


parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=10, help="number of classes")
parser.add_argument("--model_type", type=str, default='vit', help="the model used", choices=['vit','mobile-vit'])
parser.add_argument("--model_size", type=str, default='base', help='size of vit model')

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'bce'])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=50, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

args = parser.parse_args()
args.loss_fn = torch.nn.CrossEntropyLoss()
audio_conf = {'num_mel_bins':128, 'target_length':3072, 'mean':args.dataset_mean, 'std':args.dataset_std, 'freqm':args.freqm, 'timem':args.timem}
print("audio_conf", audio_conf)
# transformer based model
train_loader = torch.utils.data.DataLoader(
            dataloader.Dataset(args.data_train, audio_conf, label_csv=args.label_csv),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        dataloader.Dataset(args.data_val, audio_conf, label_csv=args.label_csv),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

audio_model = vit_model(inp_dim=(128, 3072), patch_size=(16, 48), n_class=args.n_class, model_type=args.model_type, model_size=args.model_size)
audio_model = torch.nn.DataParallel(audio_model)
#print("\nCreating experiment directory: %s" % args.exp_dir)
#os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
#train(audio_model, train_loader, val_loader, args)

# evaluate the best model on validation set on the test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
audio_model.load_state_dict(sd)

# best model on the validation set
stats, _ = validate(audio_model, val_loader, args, 'valid_set')
# note it is NOT mean of class-wise accuracy
val_acc = stats[0]['acc']
val_mAUC = np.mean([stat['auc'] for stat in stats])
val_mAP = np.mean([stat['AP'] for stat in stats])
print('---------------evaluate on the validation set---------------')
print("Accuracy: {:.6f}".format(val_acc))
print("AUC: {:.6f}".format(val_mAUC))
print("mAP: {:.6f}".format(val_mAP))

# test the model on the evaluation set
eval_loader = torch.utils.data.DataLoader(
    dataloader.Dataset(args.data_eval, audio_conf, label_csv=args.label_csv),
    batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
eval_acc = stats[0]['acc']
eval_mAUC = np.mean([stat['auc'] for stat in stats])
eval_mAP = np.mean([stat['AP'] for stat in stats])
print('---------------evaluate on the test set---------------')
print("Accuracy: {:.6f}".format(eval_acc))
print("AUC: {:.6f}".format(eval_mAUC))
print("mAP: {:.6f}".format(eval_mAP))
with open("%s/test_stats_final.pkl" %args.exp_dir, "wb") as f:
    pickle.dump(stats, f)
np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])


