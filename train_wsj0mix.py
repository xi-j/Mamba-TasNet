#!/usr/bin/env/python3

'''
Copied and modified from
https://github.com/speechbrain/speechbrain/blob/develop/recipes/WSJ0Mix/separation/train.py
'''


"""Recipe for training a neural speech separation system on wsjmix the
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import wandb
import logging
from speechbrain.core import AMPConfig

import pprint
pp = pprint.PrettyPrinter(indent=4)

from speechbrain.utils.logger import format_order_of_magnitude

os.environ['WANDB__SERVICE_WAIT'] = '999999'

def check_mean_max_grad(parameters):

    # Calculate gradient norm (L2 norm as an example)
    grad_norm = torch.sqrt(sum(p.grad.data.norm(2) ** 2 for p in parameters if p.grad is not None))

    # Calculate maximum gradient
    grad_max = max(p.grad.data.abs().max() for p in parameters if p.grad is not None)

    return grad_norm, grad_max


# Define training procedure
class Separation(sb.Brain):

    def train_augment(self, mix, targets, mix_lens):
        # Add speech distortions
        with torch.no_grad():
            if self.hparams.use_speedperturb:
                mix, targets = self.add_speed_perturb(targets, mix_lens)
                mix = targets.sum(-1)

            if self.hparams.use_wavedrop:
                mix = self.hparams.drop_chunk(mix, mix_lens)
                mix = self.hparams.drop_freq(mix)

            if self.hparams.limit_training_signal_len:
                mix, targets = self.cut_signals(mix, targets)

        return mix, targets

    def compute_forward(self, mix, stage):
        """Forward computations from the mixture to the separated signals."""
        # Separation
        mix_w = self.hparams.Encoder(mix)
        est_mask = self.hparams.MaskNet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.hparams.Decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :] # (B, T, S)

        return est_source

    def compute_objectives(self, predictions, targets, mix=None):
        """Computes the sinr loss"""
        loss = self.hparams.loss(targets, predictions)

        if mix != None:
            with torch.no_grad():
                mix = torch.stack(
                    [mix] * self.hparams.num_spks, dim=-1
                )
                sisnr_baseline = self.compute_objectives(
                    mix, targets
                )
                sisnr_i = loss - sisnr_baseline
                numel = loss.numel()
                nan_mask = torch.isnan(loss)
                nan_ratio = nan_mask.sum().item()
                sisnr_i = sisnr_i[~nan_mask].mean()
                sisnr = loss[~nan_mask].mean()
            
                self.loss_stat['nan-ratio'] += nan_ratio
                self.loss_stat['-si-snr'] += numel * sisnr
                self.loss_stat['-si-snri'] += numel * sisnr_i
                self.count += numel

        return loss

    def fit_batch(self, batch):
        """Trains one batch"""
        should_step = (self.step % self.grad_accumulation_factor) == 0

        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mixture
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)
    
        with self.no_sync(not should_step):
            if self.use_amp:
                with torch.autocast(
                    dtype=torch.bfloat16, device_type=torch.device(self.device).type,
                ):
                    mix, targets = self.train_augment(mix, targets, mix_lens)
                    predictions = self.compute_forward(mix, sb.Stage.TRAIN)

                    if hasattr(self.hparams, 'clip_wav_max'):
                        predictions = torch.clamp(predictions, 
                            min=-self.hparams.clip_wav_max, 
                            max=self.hparams.clip_wav_max
                        )

                    loss = self.compute_objectives(predictions, targets, mix)

                    # hard threshold the easy dataitems
                    if self.hparams.threshold_byloss:
                        th = self.hparams.threshold
                        loss = loss[loss > th]
                        if loss.nelement() > 0:
                            loss = loss.mean()
                    else:
                        loss = loss.mean()

                if (
                    loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
                ):  # the fix for computational problems
                    self.scaler.scale(loss).backward()
                    if self.hparams.clip_grad_norm >= 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.modules.parameters(),
                            self.hparams.clip_grad_norm,
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    '''
                    grad_norm, grad_max = check_mean_max_grad(
                        self.optimizer.param_groups[0]['params']
                    )
                    est_max = (
                        torch.max(predictions[..., 0]) \
                        + torch.max(predictions[..., 1])
                    )/2
                    est_norm = (
                        torch.norm(predictions[..., 0], p=2) \
                        + torch.norm(predictions[..., 1], p=2)
                    )/2
                    tar_max = (
                        torch.max(targets[..., 0]) \
                        + torch.max(targets[..., 1])
                    )/2
                    tar_norm = (
                        torch.norm(targets[..., 0], p=2) \
                        + torch.norm(targets[..., 1], p=2)
                    )/2

                    self.hparams.train_logger.run.log(
                        data={
                            'train.grad_norm': float(grad_norm),
                            'train.grad_max': float(grad_max),
                            'train.step_loss': float(loss),
                            'train.est_max': float(est_max),
                            'train.est_norm': float(est_norm),
                            'train.est2tar_max': float(est_max/tar_max+1e-8),
                            'train.est2tar_norm': float(est_norm/tar_norm+1e-8),
                        },
                        step=self.global_fit_step
                    )
                    '''


                else:
                    self.nonfinite_count += 1
                    logger.info(
                        "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                            self.nonfinite_count
                        )
                    )
                    loss.data = torch.tensor(0).float().to(self.device)
            else:
                mix, targets = self.train_augment(mix, targets, mix_lens)
                predictions = self.compute_forward(mix, sb.Stage.TRAIN)
                loss = self.compute_objectives(predictions, targets, mix)

                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss = loss[loss > th]
                    if loss.nelement() > 0:
                        loss = loss.mean()
                else:
                    loss = loss.mean()

                if (
                    loss.nelement() > 0 and loss < self.hparams.loss_upper_lim
                ):  # the fix for computational problems
                    loss.backward()
                    if self.hparams.clip_grad_norm >= 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.modules.parameters(),
                            self.hparams.clip_grad_norm,
                        )
                    self.optimizer.step()
                else:
                    self.nonfinite_count += 1
                    logger.info(
                        "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                            self.nonfinite_count
                        )
                    )
                    loss.data = torch.tensor(0).float().to(self.device)

        self.optimizer.zero_grad()

        if hasattr(self, "warmup_scheduler") and \
            self.global_fit_step < self.hparams.n_warmup_step:
            self.warmup_scheduler.step()
        elif isinstance(
            self.hparams.lr_scheduler, CosineAnnealingLR
        ):
            self.hparams.lr_scheduler.step()
        self.global_fit_step += 1

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mixture
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        with torch.no_grad():
            predictions = self.compute_forward(mix, stage)
            loss = self.compute_objectives(predictions, targets, mix)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.mean().detach()

    def on_fit_start(self):
        self._compile()
        self._wrap_distributed()
        self.init_optimizers()

    def init_optimizers(self):

        super().init_optimizers()
        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible()

        cur_epoch = self.hparams.epoch_counter.current
        n_step_per_epoch = 20000 // self.hparams.batch_size
        cur_step = cur_epoch * n_step_per_epoch
        self.global_fit_step = cur_step
        print(f'Start training at {str(cur_step)} steps.')

        if hasattr(self.hparams, "n_warmup_step") and \
            self.hparams.n_warmup_step != 0: # Linear warmup
            from transformers import get_constant_schedule_with_warmup
            self.warmup_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.hparams.n_warmup_step
            )
            print(f'Linear warmup for {str(self.hparams.n_warmup_step)} steps.')

        if hasattr(self.hparams, "use_cosine_schedule") and self.hparams.use_cosine_schedule: # override lr scheduler
            self.hparams.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=n_step_per_epoch*self.hparams.N_epochs-self.hparams.n_warmup_step, 
                eta_min=self.hparams.lr*self.hparams.percent_lr,
                last_epoch=cur_step,
                verbose=False
            )
            print('Use CosineAnnealingLR scheduler.')
            self.hparams.lr_scheduler._step_count = cur_step + 1
            self.optimizer._step_count = cur_step

        else:
            print(self.hparams.lr_scheduler)

    def on_stage_start(self, stage, epoch=None):
        self.loss_stat = {
            'nan-ratio': 0,
            '-si-snr': 0,
            '-si-snri': 0,
        }
        self.count = 0

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": round(stage_loss, 2)}
        for key in self.loss_stat.keys():
            stage_stats[key] = round(float(self.loss_stat[key]/self.count), 2)

        if stage == sb.Stage.TRAIN:
            self.train_stats = copy.deepcopy(stage_stats)
            print(f'Epoch {str(epoch)} TRAIN:')
            pp.pprint(stage_stats)

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # ReduceLROnPlateau
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)

            # IntervalScheduler
            elif isinstance(
                self.hparams.lr_scheduler, schedulers.IntervalScheduler
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(self.optimizer)
                print(self.optimizer.param_groups[0]["lr"])

            else:
                # no change
                current_lr = self.optimizer.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"-si-snr": stage_stats["-si-snr"]}, min_keys=["-si-snr"],
            )

            print(f'Epoch {str(epoch)} VALID:')
            pp.pprint(stage_stats)

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            print('TEST:')
            pp.pprint(stage_stats)


    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speed_perturb(targets[:, :, i])
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, batch_size=1
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mix, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    mix, mix_len = mix.to(self.device), mix_len.to(self.device)

                    # Convert targets to tensor
                    targets = torch.cat(
                        [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
                        dim=-1,
                    ).to(self.device)

                    with torch.no_grad():
                        predictions = self.compute_forward(mix, sb.Stage.TEST)

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mix] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Provide audio pipelines

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        return mix_sig

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        return s1_sig

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        s2_sig = sb.dataio.dataio.read_audio(s2_wav)
        return s2_sig

    if hparams["num_spks"] == 3:

        @sb.utils.data_pipeline.takes("s3_wav")
        @sb.utils.data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            s3_sig = sb.dataio.dataio.read_audio(s3_wav)
            return s3_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    if hparams["num_spks"] == 3:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
        )

    return train_data, valid_data, test_data


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Check if wsj0_tr is set with dynamic mixing
    if hparams["dynamic_mixing"] and not os.path.exists(
        hparams["base_folder_dm"]
    ):
        raise ValueError(
            "Please, specify a valid base_folder_dm folder when using dynamic mixing"
        )

    # Data preparation
    from utils.prepare_data import prepare_wsjmix  # noqa

    run_on_main(
        prepare_wsjmix,
        kwargs={
            "datapath": hparams["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": hparams["skip_prep"],
            "fs": hparams["sample_rate"],
        },
    )

    # Create dataset objects
    if hparams["dynamic_mixing"]:
        from utils.dynamic_mixing import dynamic_mix_data_prep

        # if the base_folder for dm is not processed, preprocess them
        if "processed" not in hparams["base_folder_dm"]:
            # if the processed folder already exists we just use it otherwise we do the preprocessing
            if not os.path.exists(
                os.path.normpath(hparams["base_folder_dm"]) + "_processed"
            ):
                from utils.preprocess_dynamic_mixing import resample_folder

                print("Resampling the base folder")
                run_on_main(
                    resample_folder,
                    kwargs={
                        "input_folder": hparams["base_folder_dm"],
                        "output_folder": os.path.normpath(
                            hparams["base_folder_dm"]
                        )
                        + "_processed",
                        "fs": hparams["sample_rate"],
                        "regex": "**/*.wav",
                    },
                )
                # adjust the base_folder_dm path
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )
            else:
                print(
                    "Using the existing processed folder on the same directory as base_folder_dm"
                )
                hparams["base_folder_dm"] = (
                    os.path.normpath(hparams["base_folder_dm"]) + "_processed"
                )

        # Colleting the hparams for dynamic batching
        dm_hparams = {
            "train_data": hparams["train_data"],
            "data_folder": hparams["data_folder"],
            "base_folder_dm": hparams["base_folder_dm"],
            "sample_rate": hparams["sample_rate"],
            "num_spks": hparams["num_spks"],
            "training_signal_len": hparams["training_signal_len"],
            "dataloader_opts": hparams["dataloader_opts"],
        }
        train_data = dynamic_mix_data_prep(dm_hparams)
        _, valid_data, test_data = dataio_prep(hparams)
    else:
        train_data, valid_data, test_data = dataio_prep(hparams)

    # Initialize wandb

    if hparams['use_wandb']:
        hparams['train_logger'] = hparams['wandb_logger']()

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    # Training
    if not hparams['eval_only']:
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    separator.evaluate(test_data, min_key="-si-snr")
    separator.save_results(test_data)

    wandb.finish()
