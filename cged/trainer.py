import torch, random, gc
from torch import nn, optim
from tqdm import tqdm
from transformers import AdamW
from transformers.models.bart.modeling_bart import shift_tokens_right
import os


class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val if val else 1e-5
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """
        String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class Trainer(nn.Module):
    def __init__(self, model, data, args, device, tokenizer):
        super().__init__()
        self.args = args
        self.model = model
        self.config = model.model.config
        self.data = data
        self.device = device
        self.device_id = args.device_ids[0]
        self.tokenizer = tokenizer
        self.valid_epoch = 0  # 记录验证集loss不下降的次数
        self.start_epoch = 0  # 开始的epoch

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'AdamW':
            self.optimizer = AdamW(model.parameters())
        else:
            raise Exception("Invalid optimizer.")

        if args.visible_gpu > -1:
            self.cuda(device=self.device_id)
        self.bt_loss = float('inf')

    def text_batching(self, batch):
        tgt_labels, _src_lines = [], []
        input_ids, attention_mask = [], []
        decoder_attention_mask = []
        _max_src_len, _max_tgt_len = 0, 0

        for b in batch:
            _src_lines.append(b['src'])
            _src_len = len(b['src']) + 2
            _tgt_len = len(b['tgt']) + 1
            _max_src_len = max(_max_src_len, _src_len)
            _max_tgt_len = max(_max_tgt_len, _tgt_len)  # cls

        for src_line, b in zip(_src_lines, batch):
            src_input_ids = [self.tokenizer.cls_token_id] + src_line + [self.tokenizer.sep_token_id]
            src_input_len = len(src_input_ids)  # valid length

            src_input_ids = src_input_ids + [self.tokenizer.pad_token_id] * (_max_src_len - src_input_len)
            input_ids.append(torch.tensor(src_input_ids).unsqueeze(0))

            _att_mask = [1] * src_input_len + [0] * (_max_src_len - src_input_len)
            attention_mask.append(torch.tensor(_att_mask).unsqueeze(0))

            # only need sep token
            _tgt_input_ids = b['tgt'] + [self.tokenizer.sep_token_id]
            tgt_input_len = len(_tgt_input_ids)
            _tgt_input_ids = _tgt_input_ids + [self.tokenizer.pad_token_id] * (_max_tgt_len - tgt_input_len)
            tgt_labels.append(torch.tensor(_tgt_input_ids).unsqueeze(0))

            _dec_att_mask = [1] * (tgt_input_len) + [0] * (_max_tgt_len - tgt_input_len)
            decoder_attention_mask.append(torch.tensor(_dec_att_mask).unsqueeze(0))

        input_ids = torch.cat(input_ids).long()
        attention_mask = torch.cat(attention_mask).long()
        tgt_labels = torch.cat(tgt_labels).long()

        # shift_tokens_right
        decoder_input_ids = shift_tokens_right(tgt_labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        decoder_attention_mask = torch.cat(decoder_attention_mask).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tgt_labels": tgt_labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask
        }

    def train_model(self, resume: bool = False):
        train_loader = self.data.train_loader
        train_num = len(train_loader)
        batch_size = self.args.batch_size
        total_batch = train_num // batch_size + 1
        start_metric_epoch = 0

        for epoch in tqdm(range(self.start_epoch, self.args.max_epoch), desc="training"):
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)

            print("\n=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()

            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                train_instance = train_loader[start:end]
                if not train_instance:
                    continue

                tensor_ori = self.text_batching(train_instance)

                logits = self.model(input_ids=tensor_ori["input_ids"].cuda(device=self.device_id),
                                    attention_mask=tensor_ori["attention_mask"].cuda(device=self.device_id),
                                    decoder_input_ids=tensor_ori["decoder_input_ids"].cuda(device=self.device_id),
                                    labels=tensor_ori["tgt_labels"].cuda(device=self.device_id),
                                    decoder_attention_mask=tensor_ori["decoder_attention_mask"].cuda(
                                        self.device_id))

                avg_loss.update(logits.loss.item(), 1)
                logits.loss.backward()

                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()

                if (batch_id + 1) % self.args.print_steps == 0:
                    print()
                    print("Instance: %d; overall loss: %.4f" % (start, avg_loss.avg), flush=True)
                    gc.collect()

            # early stopping
            if self.valid_epoch > self.args.valid_max_epoch:
                print(f'early stop training at epoch {epoch}')
                break

            # validation
            print("===== Epoch %d Validation =====" % epoch)
            res = self.evaluate_valid(epoch, start_metric_epoch)
            gc.collect()
            torch.cuda.empty_cache()

            if not res:
                print(f'early stop training at epoch {epoch}')
                break

    def evaluate_valid(self, epoch, start_metric_epoch):
        valid_loss = self.eval_model(self.data.valid_loader, epoch)  # result
        print("valid loss: %.4f" % valid_loss.avg, flush=True)

        if valid_loss.avg < self.bt_loss:
            self.valid_epoch = 0
            self.bt_loss = valid_loss.avg
            best_result_epoch = epoch

            if not os.path.exists(self.args.generated_param_directory):
                os.makedirs(self.args.generated_param_directory)

            ### model name need to change
            print('\n' + '=' * 50)
            print('saved best weighted epoch:{}'.format(epoch))
            print('best loss:{}'.format(self.bt_loss))
            # update model name
            _model_name = os.path.join(self.args.generated_param_directory, self.args.model_name + '.pt')

            torch.save({'epoch': best_result_epoch,
                        # 'optimizer': self.optimizer.state_dict(),  # saved optimizer
                        'state_dict': self.model.state_dict(),
                        'bt_loss': self.bt_loss},
                       _model_name)
        else:
            self.valid_epoch += 1

        # 停止训练
        if self.valid_epoch > self.args.valid_max_epoch:
            print(f'early stop training at epoch {epoch}')
            return False

        return True

    def eval_model(self, eval_loader, epoch: int):
        self.model.eval()
        avg_loss = AverageMeter()
        with torch.no_grad():
            batch_size = self.args.valid_batch_size
            eval_num = len(eval_loader)
            total_batch = eval_num // batch_size + 1
            for batch_id in tqdm(range(total_batch), desc="evaluating"):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > eval_num:
                    end = eval_num
                eval_instance = eval_loader[start:end]

                if not eval_instance:
                    continue
                tensor_ori = self.text_batching(eval_instance)

                if epoch > -1:
                    logits = self.model(input_ids=tensor_ori["input_ids"].cuda(device=self.device_id),
                                        attention_mask=tensor_ori["attention_mask"].cuda(device=self.device_id),
                                        decoder_input_ids=tensor_ori["decoder_input_ids"].cuda(device=self.device_id),
                                        decoder_attention_mask=tensor_ori["decoder_attention_mask"].cuda(
                                            device=self.device_id),
                                        labels=tensor_ori["tgt_labels"].cuda(device=self.device_id))

                    avg_loss.update(logits.loss.item(), 1)
                    if batch_id % self.args.print_steps == 0:
                        print()
                        print("Ins: %d; overall loss: %.4f" % (start, avg_loss.avg), flush=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    # produce target_sentence
                    preds = self.model.generate(tensor_ori["input_ids"].cuda(device=self.device_id),
                                                num_beams=self.args.beam_search,
                                                max_length=self.args.max_length,
                                                early_stopping=True)
                    predictions = [self.tokenizer.decode(g, skip_special_tokens=True) for g in preds]
                    self.prediction_list += predictions
        return avg_loss

    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        self.bt_loss = checkpoint['bt_loss']

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer
