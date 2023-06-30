import os
import time
import shutil

import torch
import numpy

import data
from model import SAEM
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
from scipy.spatial.distance import cdist

import logging
from datetime import datetime
import tensorboard_logger as tb_logger

import argparse
import parts
import time

current_path = '/home/IT_matching/matching/sh_code/sh_newcode/expand+attention'


def main():
    # Hyper Parameterss
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/IT_matching/matching/data',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_bottom',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=40, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', default=True, action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--final_dims', default=256, type=int,
                        help='dimension of final codes.')
    parser.add_argument('--max_words', default=32, type=int,
                        help='maximum number of words in a sentence.')
    parser.add_argument("--bert_path",
                        default='/home/IT_matching/matching/data/bert/uncased_L-12_H-768_A-12/',
                        type=str,
                        help="The BERT model path.")
    parser.add_argument("--txt_stru", default='cnn',
                        help="Whether to use pooling or cnn or rnn")
    parser.add_argument("--trans_cfg", default='t_cfg.json',
                        help="config file for image transformer")

    opt = parser.parse_args()

    opt.model_name = current_path

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    opt.logger_name = opt.logger_name + TIMESTAMP
    tb_logger.configure(opt.logger_name, flush_secs=5)

    f = open(opt.logger_name + "opt.txt", 'w')
    f.write(opt.__str__())
    f.close()

    opt.vocab_file = opt.bert_path + 'vocab.txt'
    opt.bert_config_file = opt.bert_path + 'bert_config.json'
    opt.init_checkpoint = opt.bert_path + 'pytorch_model.bin'
    opt.do_lower_case = True

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, opt.batch_size, opt.workers, opt)
    test_loader = data.get_test_loader('test',
                                       opt.data_name, opt.batch_size, opt.workers, opt)
    # Construct the model
    model = SAEM(opt)

    localtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = os.path.join(current_path, 'model_backup/{}_{}_{}_{}_{}'.format(opt.data_name.split('_')[0], opt.batch_size,
                                                                           opt.embed_size, opt.word_dim, localtime))
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = os.path.join(path, '{}_{}'.format(opt.data_name,
                                                   opt.batch_size) + '.pth.tar')

    best_rsum = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(False, opt, val_loader, model, path)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum = validate(False, opt, test_loader, model, path)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        if is_best:
            validate(True, opt, test_loader, model, path)

        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), model_path=model_path)


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(epoch, train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        # if model.Eiters % opt.val_step == 0:
        #     validate(opt, val_loader, model)


def write_txt(save_path, rsum, ar, r, ari, ri):
    txt = []
    txt.append('------------------------------------------')
    txt.append("rsum: %.1f" % rsum)
    txt.append("Average i2t Recall: %.1f" % ar)
    txt.append("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    txt.append("Average t2i Recall: %.1f" % ari)
    txt.append("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    txt.append('------------------------------------------')
    # current_path = opt.current_path
    file_name = 'result.txt'
    parts.write_txt_add(os.path.join(save_path, file_name), txt)


def validate(issave, opt, val_loader, model, path=''):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs, cap_lens = encode_data(
        model, val_loader, opt, opt.log_step, logging.info)

    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    start = time.time()
    sims = 1 - cdist(img_embs, cap_embs, metric='cosine')
    end = time.time()
    print("calculate similarity time:", end - start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, cap_lens, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    if issave:
        print('Best Model parameters Update!')
        r, ri = (r1, r5, r10, medr, meanr), (r1i, r5i, r10i, medri, meanr)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        numpy.save(os.path.join(path, 'FinalSims_{}.npy'.format(opt.data_name)), sims)
        write_txt(path, currscore, ar, r, ari, ri)
        return 0

    print(currscore)

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', model_path=''):
    tries = 15
    error = None
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            # torch.save(state, prefix + filename)
            if is_best:
                # shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
                torch.save(state, model_path)
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""

    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
