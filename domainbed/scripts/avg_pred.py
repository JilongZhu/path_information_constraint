# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import copy

import numpy as np
import PIL
import torch
import torchvision
import torch.nn as nn
import torch.utils.data

from domainbed.lib.query import Q 
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--trade_off', type=float, default=1, help="parameter for refer loss")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    # print("Environment:")
    # print("\tPython: {}".format(sys.version.split(" ")[0]))
    # print("\tPyTorch: {}".format(torch.__version__))
    # print("\tTorchvision: {}".format(torchvision.__version__))
    # print("\tCUDA: {}".format(torch.version.cuda))
    # print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    # print("\tNumPy: {}".format(np.__version__))
    # print("\tPIL: {}".format(PIL.__version__))

    # print('Args:')
    # for k, v in sorted(vars(args).items()):
    #     print('\t{}: {}'.format(k, v))

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
    from IPython import embed;embed();exit();
    # seed 0
    # unique_filenames = {'PACS' : [['433548cd-c782-48e2-a7f4-205d15ccd668'],['fc2aa9a1-b6e1-4051-b70f-d48e62acdb3b'],['0ce5d602-502c-4cb3-be92-4f7f741fceab'],['b21d4414-8568-4ebc-a17e-9c2d76131bd2']],
    #                     'VLCS' : [['fc466f4e-1e09-4a7e-b6e5-fdce53d4c0b5'],['fac95d17-63f5-4d10-a703-460be2a49368'],['b0683227-1388-4f42-954f-40d2a01f1415'],['f05a0baf-752f-482d-a972-68e3d930a02d']],
    #                     'OfficeHome' : [['bcd45379-3454-4e72-abfe-f32db18c1b5a'],['f0d355e3-fdbd-4c7c-a6db-fbce97e5e314'],['63737763-70c4-42a4-bb1b-76bdcd6e534c'],['189a0730-7741-4f81-bec5-62b9d617752c']],
    #                     'TerraIncognita' : [['e76f1d0c-fdae-4280-9e00-6befa5706f46'],['f4653a63-8bfe-4267-884e-2a6784b0ed3f'],['2972e1e4-0619-4b91-9056-dd8c242c4a1d'],['84ab8715-d398-4b5a-8f26-1ff00d3ea3b2']]}
    # seed 9999
    # unique_filenames = {'PACS' : [['18eb3479-abc9-40ae-b4af-a6b62d79c7f3'],['505d844f-3fe2-43ca-b7e7-8c7cc2f46f36'],['8934e186-4771-495c-9b12-212db1831cb0'],['a66549ed-b798-4875-a503-0096c69cf3e4']],
    #                     'VLCS' : [['16d8f651-e738-48e4-835b-ea53d6f862d8'],['6c9cf251-7d70-480b-87ab-2ad8dc575af6'],['10c87581-ad77-4966-a3ad-b33561dcdaf4'],['8c6bb81d-b53d-417c-be59-52ccaa23a7f8']],
    #                     'OfficeHome' : [['6c9dff6a-da4f-41df-8fa3-3747e6b772db'],['4cdb0711-1cd5-4628-8d29-34422ac1bf9f'],['d9eb8596-5983-43bf-ac74-b40c3f05da44'],['c6de4620-01fe-4e87-97d3-7d6a802fb6c1']],
    #                     'TerraIncognita' : [['b1b37537-8192-45f1-b539-edfdcf758373'],['a110a651-86ad-4a66-a13a-111d57f57eab'],['4cb83548-81f1-4c89-9657-867adf6a06e4'],['4b7e68df-df32-4374-83ab-0b8521726174']]}
    # seed 6999
    # unique_filenames = {'PACS' : [['2d602c43-ad29-4c7c-990e-58283c97f5dc'],['436c627d-d7e6-468a-9c43-2ad4b146b892'],['2b2c2111-6d1f-4a83-b8cb-4c4ef9a8a579'],['a2ce4a14-c38a-4240-8ad8-bec9705b11e9']],
    #                     'VLCS' : [['12d9360e-8829-42b7-a810-730e139c7a09'],['44d236cf-2279-4c9f-90bb-fcfc40224a34'],['cdbcca1e-aabb-42fe-bd62-4efde8c6882d'],['fb79d1fe-d3d7-4f9a-b066-f9bd47b35c56']],
    #                     'OfficeHome' : [['7d0befca-505b-4a15-b8a8-051e72e725be'],['174910a8-d16b-4ba0-9e33-a6021c01a55b'],['b5c1e532-4eaa-4c4a-bf53-fe3435d6e443'],['bae271b4-beda-4f66-9f62-a7a779ecf1c4']],
    #                     'TerraIncognita' : [['f1f0ab5f-04b6-4ad0-b0fc-cdc670558838'],['6ec7bb5f-2e55-4df4-bac4-598812d4a425'],['ac14c3e7-5f66-4a7e-80ec-224682e2f7d5'],['ffcfcb8d-2f0f-4399-bd80-3d42f6b9306d']]}
    # seed 3999
    # unique_filenames = {'PACS' : [['5ba0aab9-f093-4b81-9af1-db31a6888a41'],['2ccf90d5-6952-4441-b3ed-77a479f95fc6'],['7ca4c98f-89e8-42b7-82e8-dc210c9a87de'],['82154b1e-4ed5-4a11-99f6-714e456129f4']],
    #                     'VLCS' : [['9cf86686-243d-498b-95b1-7588d8fc0b00'],['9eb1bf19-e536-4b26-842f-b76264a8f89c'],['1c0b10d4-0c4f-4ed8-b341-8edeeb7aaf9a'],['fcfb455c-2989-441f-bd11-109e2a6dd501']],
    #                     'OfficeHome' : [['2c02c9c2-f5f2-4860-b7cd-afe029c0e077'],['3df552cb-48e0-4381-82d2-0daaeaca3454'],['1dfb8cd0-3c2e-4cfd-8532-e57ba78bc95d'],['cbb44ef9-973e-48ec-b923-3574c3f6db42']],
    #                     'TerraIncognita' : [['56e94398-0e7b-42c5-a664-add3ad1676ab'],['68c94a4f-4a7a-4f16-9505-50b1ea2a16df'],['27afee80-43cf-40a3-8d62-7efee9837af3'],['f77cc509-8f68-490a-9aad-742d33b18a58']]}
    # seed 1999
    unique_filenames = {'PACS' : [['6fb75047-6f08-4d3d-943a-5fa6aa0dcd49'],['48efa6c7-2efd-46a4-9165-eeeb842e9f1f'],['66a127e5-290b-4068-b3ac-ffcaf07df267'],['2b92a197-5a14-41a1-ab79-f2974757399e']],
                        'VLCS' : [['950e1898-aef3-4ec7-a8f6-12292ef89266'],['a734a3f3-db31-4f00-9c28-6ea7f2a56d36'],['1dfa106d-f015-4bd7-95fc-f1663783080a'],['d8c7e376-2b9e-48c3-8547-388381900a63']],
                        'OfficeHome' : [['aca5d8aa-f649-4d38-93d2-c5d194d4c4f8'],['167661cd-c1f8-4785-9f1a-b8d254c0ee53'],['4a4cc3bc-51d5-4a0b-a86d-5bf5d7d667f3'],['aca0ba70-41e5-4e75-b440-a4c625295a06']],
                        'TerraIncognita' : [['5d11cb0f-61b9-4507-9d6f-1b6d2ba3f746'],['fe4b8913-7f4b-44ab-97b7-3cdd0ec17de6'],['1c6a2258-6f58-47f9-80ef-08670ccf20ef'],['ccbf9b05-82db-4e3d-ae2d-ecae1f1b619f']]}



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

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
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

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    # train_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(in_splits)
    #     if i not in args.test_envs]

    # uda_loaders = [InfiniteDataLoader(
    #     dataset=env,
    #     weights=env_weights,
    #     batch_size=hparams['batch_size'],
    #     num_workers=dataset.N_WORKERS)
    #     for i, (env, env_weights) in enumerate(uda_splits)
    #     if i in args.test_envs]

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
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": model
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    # aver_model = []
    for unique_filename in unique_filenames[args.dataset][args.test_envs[0]]:
        best_models = []
        print(unique_filename)
        for iters in range(n_domains):
            best_models.append(torch.load(os.path.join(args.output_dir,unique_filename + f'_path_{iters}.pkl'))['model_dict']) 
        weight_aver_model = misc.get_weight_aver_model(best_models)
        # aver_model.append(weight_aver_model)
        fname = args.dataset+'_'+str(args.test_envs[0])+'_'+unique_filename+'.pkl'
        print(fname)
        save_checkpoint(weight_aver_model, fname)
        

    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    for name, loader, weights in evals:
        env_name, inout = name.split("_")
        env_num = int(env_name[3:])
        is_test = env_num in args.test_envs
        if is_test:
            pred_aver = misc.accuracy(weight_aver_model,loader,weights,device)
            # pred_aver = misc.pred_aver_acc(aver_model, loader, weights, device)
            # pred_aver = misc.pred_aver_acc(best_models, loader, weights, device)
            # weight_aver = misc.weight_aver_acc(best_models, loader, weights, device)
            # weight_aver_model = misc.get_weight_aver_model(best_models)
            # weight_aver2 = misc.accuracy(weight_aver_model,loader,weights,device)
            # print(name+' pred_aver_acc: {:.5f}, weight_aver_acc: {:.5f}, weight_aver_acc2: {:.5f}'.format(pred_aver, weight_aver, weight_aver2))
            print(name+' pred_aver_acc: {:.5f}'.format(pred_aver))


#    save_checkpoint('model.pkl')

#    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
#        f.write('done')
