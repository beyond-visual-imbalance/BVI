# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task, visual_halllucination =None, src_img_features= None):
        super().__init__(args, task)
        self.eps = args.label_smoothing
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model=None, sample=None, visual_hallucination=None, src_img_features=None,reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if sample['net_input'].get('src_img_features') is not None:
            # 收集所有的pair[0]和pair[1]
            all_pair_0_tensor = [torch.tensor(p[0].reshape(1, 49, 2048)).half().cuda() for p in sample['net_input']['src_img_features']]
            all_pair_1_tensor = [torch.tensor(p[1].reshape(1, 49, 2048)).half().cuda() for p in sample['net_input']['src_img_features']]

            all_pair_0_tensor = torch.cat(all_pair_0_tensor, dim=0)
            all_pair_1_tensor = torch.cat(all_pair_1_tensor, dim=0)


            all_pair_0_list = list(all_pair_0_tensor.unbind())
            all_pair_1_list = list(all_pair_1_tensor.unbind())


            all_pair_0_expanded = [x.unsqueeze(0) for x in all_pair_0_list]
            all_pair_1_expanded = [x.unsqueeze(0) for x in all_pair_1_list]


            src_img_features = list(zip(all_pair_0_expanded, all_pair_1_expanded))


            sample['net_input']['src_img_features'] = src_img_features




            net_output = model(**sample['net_input'])
        else:
            net_output = model(**sample['net_input'])



        loss, nll_loss = self.compute_loss(model, net_output[0], sample, reduce=reduce)


        if net_output[1].encoder_out is not None:
            if net_output[1].src_img_features is not None:
                kl_divergence = self.kl_divergence_loss(net_output[1].hallucination_train.float().cpu(), net_output[1].src_img_features.float().cpu(), net_output[1].src_img_features.size(-1), net_output[1].hallucination_train.size(-1))

                loss = loss.to(device) + 15*kl_divergence

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss


    def kl_divergence_loss(self, visual_hallucination, src_img_features, dim, dim_2, epsilon=1e-10):


        fc = nn.Linear(dim, dim_2)

        src_img_features = fc(src_img_features)


        visual_hallucination_prob = F.softmax(visual_hallucination, dim=-1) + epsilon
        src_img_features_prob = F.softmax(src_img_features, dim=-1) + epsilon


        log_visual_hallucination_prob = torch.log(visual_hallucination_prob)


        kl_div = F.kl_div(log_visual_hallucination_prob.to(device), src_img_features_prob.to(device), reduction='batchmean')

        return kl_div




    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['nll_loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


def tensor_statistics(tensor, name="Tensor"):
    print(f"--- {name} Statistics ---")
    print("Mean:", tensor.mean().item())
    print("Std Dev:", tensor.std().item())
    print("Max Value:", tensor.max().item())
    print("Min Value:", tensor.min().item())
    print("-------------------------\n")