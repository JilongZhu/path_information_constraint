import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.utils.data

from domainbed.lib.query import Q 
from domainbed import datasets
from domainbed import hparams_registry
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))


    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    # print('HParams:')
    # for k, v in sorted(hparams.items()):
    #     print('\t{}: {}'.format(k, v))

    print('\ndataset', args.dataset)
    print('test_envs', args.test_envs)

    unique_filenames = []
    fname  = args.dataset+'_'+str(args.test_envs[0])
    for root, _, fnames in sorted(os.walk(args.output_dir, followlinks=True)):
        for unique_fname in fnames:
            if fname in unique_fname:
                unique_filenames.append(unique_fname)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    n_domains = len(dataset) - len(args.test_envs)

    def save_checkpoint(model, filename):
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": model
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    best_models = []
    # aver_model = []
    for unique_filename in unique_filenames:
        print(unique_filename)
        # for iters in range(n_domains):
        best_models.append(torch.load(os.path.join(args.output_dir,unique_filename))['model_dict']) 
        # weight_aver_model = misc.get_weight_aver_model(best_models)
        # # aver_model.append(weight_aver_model)
        # fname = args.dataset+'_'+str(args.test_envs[0])+'_'+unique_filename+'.pkl'
        # print(fname)
        # save_checkpoint(weight_aver_model, fname)

    # print(len(best_models))    
    test_env_acc = 0
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    for name, loader, weights in evals:
        env_name, inout = name.split("_")
        env_num = int(env_name[3:])
        is_test = env_num in args.test_envs
        if is_test:
            # pred_aver = misc.accuracy(weight_aver_model,loader,weights,device)
            # pred_aver = misc.pred_aver_acc(aver_model, loader, weights, device)
            pred_aver = misc.pred_aver_acc(best_models, loader, weights, device)
            # weight_aver = misc.weight_aver_acc(best_models, loader, weights, device)
            # weight_aver_model = misc.get_weight_aver_model(best_models)
            # weight_aver2 = misc.accuracy(weight_aver_model,loader,weights,device)
            # print(name+' pred_aver_acc: {:.5f}, weight_aver_acc: {:.5f}, weight_aver_acc2: {:.5f}'.format(pred_aver, weight_aver, weight_aver2))
            print(name+' pred_aver_acc: {:.5f}'.format(pred_aver))
            if 'in' in name:
                test_env_acc += 0.8 * pred_aver
            if 'out' in name:
                test_env_acc += 0.2 * pred_aver
    
    print('test_env aver_acc: {:.5f}'.format(test_env_acc))


#    save_checkpoint('model.pkl')

#    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
#        f.write('done')
