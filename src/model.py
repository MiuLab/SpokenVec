import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from numpy.random import randint
from allennlp.modules.elmo import Elmo, batch_to_ids

import os
import glob
import time
import ipdb

import numpy as np

import modules
import model_utils

from utils import print_time_info
from vocabulary import BOS, PAD
from model_utils import build_optimizer, get_device
from modules import ELMoLM

from sklearn.metrics import classification_report
from tqdm import tqdm


class SLU:
    def __init__(self, config, model_dir, device=None):
        self.config = config
        self.model_dir = model_dir
        self.log_file = os.path.join(model_dir, 'log.csv')

        self.device = get_device(device)

        self.slu_cls = getattr(modules, config['model']['name'])
        self.slu = self.slu_cls(config['model'])

        self.use_elmo = config.get("use_elmo", False)
        if self.use_elmo:
            option_file = config["elmo"]["option_file"]
            weight_file = config["elmo"]["weight_file"]
            self.elmo = Elmo(option_file, weight_file, 1, dropout=0)
            self.slu.elmo_scalar_mixes = nn.ModuleList(self.elmo._scalar_mixes)

            if len(config["elmo"].get("checkpoint", "")) > 0:
                self.elmo._elmo_lstm = torch.load(config["elmo"]["checkpoint"]).elmo
                for param in self.elmo._elmo_lstm.parameters():
                    param.requires_grad_(False)

            self.elmo.to(self.device)

        self.slu.to(self.device)

    def prepare_training(self, batch_size, data_engine, collate_fn):
        self.train_data_loader = DataLoader(
            data_engine,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True)

        self.parameters = filter(
            lambda p: p.requires_grad, self.slu.parameters())
        self.optimizer = build_optimizer(self.config["optimizer"],
                                         self.parameters,
                                         self.config["learning_rate"])

        with open(self.log_file, 'w') as fw:
            fw.write("epoch,train_loss,train_f1,valid_loss,valid_f1,test_loss,test_f1\n")

    def prepare_testing(self, batch_size, data_engine, collate_fn):
        self.test_data_loader = DataLoader(
            data_engine,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True)

    def train(self, epochs, batch_size, data_engine,
              valid_data_engine=None, test_data_engine=None):
        collate_fn = getattr(data_engine,
                             self.config.get("collate_fn", "collate_fn_asr"))
        self.prepare_training(batch_size, data_engine, collate_fn)

        run_batch_fn = getattr(
            self, self.config.get("run_batch_fn", "run_batch"))

        for idx in range(1, epochs+1):
            epoch_loss = 0
            epoch_acc = 0.0
            batch_amount = 0

            pbar = tqdm(
                self.train_data_loader, desc="Iteration",
                ascii=True, dynamic_ncols=True)

            for b_idx, batch in enumerate(pbar):
                loss, logits = run_batch_fn(batch, testing=False)
                epoch_loss += loss.item()
                batch_amount += 1
                y_true = batch[data_engine.label_idx]
                y_pred = logits.detach().cpu().max(dim=1)[1].numpy()
                epoch_acc += (y_true == y_pred).sum() / len(y_true)
                pbar.set_postfix(
                    Loss="{:.5f}".format(epoch_loss / batch_amount),
                    Acc="{:.4f}".format(epoch_acc / batch_amount)
                )

            epoch_loss /= batch_amount
            epoch_acc /= batch_amount
            print_time_info("Epoch {} finished, training loss {}, acc {}".format(
                    idx, epoch_loss, epoch_acc))

            valid_loss, valid_acc, _, _ = self.test(batch_size, valid_data_engine)
            test_loss, test_acc = -1.0, -1.0
            if test_data_engine is not None:
                test_loss, test_acc, _, _ = self.test(batch_size, test_data_engine)
            with open(self.log_file, 'a') as fw:
                fw.write(f"{idx},{epoch_loss},{epoch_acc},"
                         f"{valid_loss},{valid_acc},{test_loss},{test_acc}\n")

            print_time_info("Epoch {}: save model...".format(idx))
            self.save_model(self.model_dir, idx)

    def test(self, batch_size, data_engine, report=False, verbose=False):
        collate_fn = getattr(
            data_engine,
            self.config.get("collate_fn_test", "collate_fn_asr"))
        self.prepare_testing(batch_size, data_engine, collate_fn)

        run_batch_fn = getattr(
            self, self.config.get("run_batch_fn", "run_batch"))

        test_probs = []
        all_y_true, all_y_pred = [], []
        test_acc = 0.0
        with torch.no_grad():
            test_loss = 0
            batch_amount = 0
            for b_idx, batch in enumerate(tqdm(self.test_data_loader)):
                loss, logits = run_batch_fn(batch, testing=True)
                test_loss += loss.item()
                batch_amount += 1
                y_true = batch[data_engine.label_idx]
                y_pred = logits.detach().cpu().max(dim=1)[1].numpy()
                test_acc += (y_true == y_pred).sum() / len(y_true)
                all_y_true += list(y_true)
                all_y_pred += list(y_pred)

            test_loss /= batch_amount
            test_acc /= batch_amount
            print_time_info("testing finished, testing loss {}, acc {}".format(
                test_loss, test_acc))

        if report:
            metrics = classification_report(
                np.array(all_y_true),
                np.array(all_y_pred),
                labels=list(range(len(data_engine.label_vocab.vocab))),
                target_names=data_engine.label_vocab.vocab,
                digits=3)
            print(metrics)

        if verbose:
            for i, (y_true, y_pred) in enumerate(zip(all_y_true, all_y_pred)):
                if y_true == y_pred:
                    continue
                label = data_engine.label_vocab.i2l(y_true)
                pred = data_engine.label_vocab.i2l(y_pred)
                print("{} [{}] [{}]".format(data_engine[i]["text"], label, pred))

        return test_loss, test_acc, all_y_true, all_y_pred

    def run_batch(self, batch, testing=False):
        if testing:
            self.slu.eval()
        else:
            self.slu.train()

        inputs, words, positions, labels = batch

        inputs = torch.from_numpy(inputs).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)

        elmo_emb = None
        if self.use_elmo:
            char_ids = batch_to_ids(words).to(self.device)
            elmo_emb = self.elmo(char_ids)['elmo_representations'][0]

        logits = self.slu(inputs, positions, elmo_emb)

        loss = F.cross_entropy(logits, labels)

        if not testing:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
            self.optimizer.step()

        return loss, logits

    def run_batch_lattice(self, batch, testing=False):
        if testing:
            self.slu.eval()
        else:
            self.slu.train()

        inputs, words, positions, prevs, nexts, labels = batch

        inputs = torch.from_numpy(inputs).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)

        elmo_emb = None
        if self.use_elmo:
            char_ids = batch_to_ids(words).to(self.device)
            elmo_emb = self.elmo(char_ids)['elmo_representations'][0]

        logits = self.slu(inputs, positions, prevs, nexts, elmo_emb)

        loss = F.cross_entropy(logits, labels)

        if not testing:
            start = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
            self.optimizer.step()
            # print(f"backward takes {time.time()-start}")

        return loss, logits

    def save_model(self, model_dir, epoch, name='slu.ckpt'):
        path = os.path.join(model_dir, "{}.{}".format(name, epoch))
        torch.save(self.slu, path)
        print_time_info("Save model successfully")

    def load_model(self, model_dir, epoch=None, name='slu.ckpt'):
        if epoch is None:
            paths = glob.glob(os.path.join(model_dir, "{}.*".format(name)))
            epoch = max(sorted(map(int,
                        [path.strip().split('.')[-1] for path in paths])))
            print_time_info("Epoch is not specified, loading the "
                            "last epoch ({}).".format(epoch))
        path = os.path.join(model_dir, "{}.{}".format(name, epoch))
        if not os.path.exists(path):
            print_time_info("Loading failed, start training from scratch...")
        else:
            self.slu.load_state_dict(
                torch.load(path, map_location=self.device).state_dict())
            if self.use_elmo and hasattr(self.slu, "elmo_scalar_mixes"):
                self.elmo._scalar_mixes = self.slu.elmo_scalar_mixes
                self.elmo.add_module('scalar_mix_0', self.elmo._scalar_mixes[0])
            print_time_info("Load model from {} successfully".format(
                model_dir))
        return epoch


class LM:
    def __init__(self, config, model_dir, device=None):
        self.config = config
        self.model_dir = model_dir
        self.log_file = os.path.join(model_dir, 'log.csv')
        self.lm_scale = config.get("lm_scale", 1.0)
        self.ca_scale = config.get("ca_scale", 0.0)
        self.n_negative_sample = config.get("n_negative_sample", 0)

        self.device = get_device(device)

        self.vocab_size = config["vocab_size"]
        option_file = config["elmo"]["option_file"]
        weight_file = config["elmo"]["weight_file"]
        self.lm = ELMoLM(option_file, weight_file, self.vocab_size)
        self.lm.to(self.device)

    def prepare_training(self, batch_size, data_engine, collate_fn):
        self.train_data_loader = DataLoader(
            data_engine,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True,
            collate_fn=collate_fn,
            pin_memory=True)

        self.parameters = filter(
            lambda p: p.requires_grad, self.lm.parameters())
        self.optimizer = build_optimizer(self.config["optimizer"],
                                         self.parameters,
                                         self.config["learning_rate"])

        with open(self.log_file, 'w') as fw:
            fw.write("epoch,train_loss,valid_loss\n")

    def prepare_testing(self, batch_size, data_engine, collate_fn):
        self.test_data_loader = DataLoader(
            data_engine,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=True)

    def train(self, epochs, batch_size, data_engine,
              valid_data_engine=None, train_decoder_epochs=0,
              max_iter_per_epoch=100000):
        collate_fn = getattr(data_engine,
                             self.config.get("collate_fn", "collate_fn"))
        self.prepare_training(batch_size, data_engine, collate_fn)

        run_batch_fn = getattr(
            self, self.config.get("run_batch_fn", "run_batch"))

        for param in self.lm.elmo.parameters():
            param.requires_grad_(False)

        for idx in range(1, epochs+1):
            if idx == train_decoder_epochs + 1 or (idx == 1 and idx > train_decoder_epochs):
                for param in self.lm.elmo.parameters():
                    param.requires_grad_(True)

            epoch_loss_for = epoch_loss_rev = epoch_loss_ca_pos = epoch_loss_ca_neg = 0
            batch_amount = 0

            pbar = tqdm(
                self.train_data_loader, desc="Iteration",
                ascii=True, dynamic_ncols=True)

            for b_idx, batch in enumerate(pbar):
                loss_for, loss_rev, loss_ca_pos, loss_ca_neg = run_batch_fn(
                    batch, testing=False)
                epoch_loss_for += loss_for.item()
                epoch_loss_rev += loss_rev.item()
                epoch_loss_ca_pos += loss_ca_pos.item()
                epoch_loss_ca_neg += loss_ca_neg.item()
                batch_amount += 1
                pbar.set_postfix(
                    FLoss="{:.5f}".format(epoch_loss_for / batch_amount),
                    BLoss="{:.5f}".format(epoch_loss_rev / batch_amount),
                    PosLoss="{:.5f}".format(epoch_loss_ca_pos / batch_amount),
                    NegLoss="{:.5f}".format(epoch_loss_ca_neg / batch_amount)
                )
                if b_idx == max_iter_per_epoch:
                    break

            epoch_loss_lm = (epoch_loss_for + epoch_loss_rev) / 2
            epoch_loss_ca = (epoch_loss_ca_pos + epoch_loss_ca_neg)
            epoch_loss = \
                (self.lm_scale * epoch_loss_lm + \
                self.ca_scale * epoch_loss_ca) / batch_amount
            print_time_info("Epoch {} finished, training loss {}".format(
                    idx, epoch_loss))

            valid_loss = self.test(batch_size, valid_data_engine)
            with open(self.log_file, 'a') as fw:
                fw.write(f"{idx},{epoch_loss},{valid_loss}\n")

            print_time_info("Epoch {}: save model...".format(idx))
            self.save_model(self.model_dir, idx)

    def test(self, batch_size, data_engine):
        collate_fn = getattr(
            data_engine,
            self.config.get("collate_fn", "collate_fn"))
        self.prepare_testing(batch_size, data_engine, collate_fn)

        run_batch_fn = getattr(
            self, self.config.get("run_batch_fn", "run_batch"))

        with torch.no_grad():
            test_loss_for = test_loss_rev = test_loss_ca_pos = test_loss_ca_neg = 0
            batch_amount = 0
            for b_idx, batch in enumerate(tqdm(self.test_data_loader)):
                loss_for, loss_rev, loss_ca_pos, loss_ca_neg = run_batch_fn(
                    batch, testing=True)
                test_loss_for += loss_for.item()
                test_loss_rev += loss_rev.item()
                test_loss_ca_pos += loss_ca_pos.item()
                test_loss_ca_neg += loss_ca_neg.item()
                batch_amount += 1

            test_loss_lm = (test_loss_for + test_loss_rev) / 2
            test_loss_ca = (test_loss_ca_pos + test_loss_ca_neg)
            test_loss = \
                (self.lm_scale * test_loss_lm + \
                self.ca_scale * test_loss_ca) / batch_amount
            print_time_info("testing finished, testing loss {}".format(test_loss))
            print_time_info(f"forward lm: {test_loss_for/batch_amount}, "
                            f"backward lm: {test_loss_rev/batch_amount}")
            print_time_info(f"ca pos: {test_loss_ca_pos/batch_amount}, "
                            f"ca neg: {test_loss_ca_neg/batch_amount}")

        return test_loss

    def run_batch(self, batch, testing=False):
        if testing:
            self.lm.eval()
        else:
            self.lm.train()

        inputs, outputs, outputs_rev, uids = batch
        char_ids = batch_to_ids(inputs).to(self.device)
        outputs = torch.from_numpy(outputs).to(self.device)
        outputs_rev = torch.from_numpy(outputs_rev).to(self.device)

        logits_forward, logits_backward, hiddens, mask = self.lm(char_ids)

        bs, sl, vs = logits_forward.size()
        logits_forward = logits_forward.view(-1, vs)
        logits_backward = logits_backward.view(-1, vs)
        outputs = outputs.view(-1)
        outputs_rev = outputs_rev.view(-1)
        loss_for = F.cross_entropy(logits_forward, outputs, ignore_index=PAD)
        loss_rev = F.cross_entropy(logits_backward, outputs_rev, ignore_index=PAD)

        loss = (loss_for + loss_rev) / 2
        if not testing:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 0.25)
            self.optimizer.step()

        return loss_for, loss_rev, torch.tensor(0.0), torch.tensor(0.0)

    def run_batch_ca(self, batch, testing=False):
        if testing:
            self.lm.eval()
        else:
            self.lm.train()

        inputs, outputs_for, outputs_rev, confs, scores = batch
        char_ids = batch_to_ids(inputs).to(self.device)
        outputs_for = torch.from_numpy(outputs_for).to(self.device)
        outputs_rev = torch.from_numpy(outputs_rev).to(self.device)

        logits_forward, logits_backward, hiddens, mask = self.lm(char_ids)

        bs, sl, vs = logits_forward.size()
        logits_forward = logits_forward.view(-1, vs)
        logits_backward = logits_backward.view(-1, vs)
        outputs_for = outputs_for.view(-1)
        outputs_rev = outputs_rev.view(-1)
        loss_for = F.cross_entropy(logits_forward, outputs_for, ignore_index=PAD)
        loss_rev = F.cross_entropy(logits_backward, outputs_rev, ignore_index=PAD)

        loss_lm = (loss_for + loss_rev) / 2

        bs_half = bs // 2
        ref_outputs = [output[:bs_half] for output in hiddens[:2]]
        hyp_outputs = [output[bs_half:] for output in hiddens[:2]]
        ref_lens = list(map(len, inputs[:bs_half]))
        hyp_lens = list(map(len, inputs[bs_half:]))

        loss_positive = torch.tensor(0.0).to(self.device)
        loss_negative = torch.tensor(0.0).to(self.device)
        y1 = torch.ones(1).to(self.device)
        y0 = torch.full((1,), -1.).to(self.device)
        denom = 0
        for b, conf in enumerate(confs):
            for ref_id, hyp_id in conf:
                ref_out = [output[b, ref_id] for output in ref_outputs]
                hyp_out = [output[b, hyp_id] for output in hyp_outputs]
                loss_positive += torch.stack([
                    F.cosine_embedding_loss(r.unsqueeze(0), h.unsqueeze(0), y1)
                    for r, h in zip(ref_out, hyp_out)
                ]).mean()

                for _ in range(self.n_negative_sample):
                    ref_out_sample = [
                        output[randint(bs_half), randint(ref_lens[b])]
                        for output in ref_outputs
                    ]
                    hyp_out_sample = [
                        output[randint(bs_half), randint(hyp_lens[b])]
                        for output in hyp_outputs
                    ]
                    loss_negative += torch.stack([
                        F.cosine_embedding_loss(r.unsqueeze(0), h.unsqueeze(0), y0)
                        for r, h in zip(ref_out_sample, hyp_out_sample)
                    ]).mean()
                denom += 1

        if denom > 0:
            loss_positive /= denom
            loss_negative /= denom
        loss_ca = loss_positive + loss_negative

        loss = self.lm_scale * loss_lm + self.ca_scale * loss_ca

        if not testing:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 0.25)
            self.optimizer.step()

        return loss_for, loss_rev, loss_positive, loss_negative

    def save_model(self, model_dir, epoch, name='lm.ckpt'):
        path = os.path.join(model_dir, "{}.{}".format(name, epoch))
        torch.save(self.lm, path)
        print_time_info("Save model successfully")

    def load_model(self, model_dir, epoch=None, name='lm.ckpt'):
        if epoch is None:
            paths = glob.glob(os.path.join(model_dir, "{}.*".format(name)))
            epoch = max(sorted(map(int,
                        [path.strip().split('.')[-1] for path in paths])))
            print_time_info("Epoch is not specified, loading the "
                            "last epoch ({}).".format(epoch))
        path = os.path.join(model_dir, "{}.{}".format(name, epoch))
        if not os.path.exists(path):
            print_time_info("Loading failed, start training from scratch...")
        else:
            self.lm.load_state_dict(
                torch.load(path, map_location=self.device).state_dict())
            print_time_info("Load model from {} successfully".format(model_dir))
        return epoch
