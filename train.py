import argparse
import glob
import json
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
from attrdict import AttrDict
from fastprogress.fastprogress import master_bar, progress_bar
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
#test
from datasets import DATASET_LIST
from model import *
from processor import seq_cls_output_modes as output_modes
from processor import seq_cls_tasks_num_labels as tasks_num_labels
from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION,
    MODEL_ORIGINER,
    init_logger,
    set_seed,
    compute_metrics
)
import inspect

logger = logging.getLogger(__name__)


def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    weight_decay_change = 'sentiment_embedding.weight'
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not n in weight_decay_change],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not n in weight_decay_change], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if n in weight_decay_change], 'weight_decay': 0.3}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))
    best_acc = 0
    acc = 0
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        ep_loss = []
        for step, (batch, txt) in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            if "KOSAC" in args.model_mode:
                inputs["polarity_ids"] = batch[4]
                inputs["intensity_ids"] = batch[5]
            if "KNU" in args.model_mode:
                inputs["polarity_ids"] = batch[4]
            if "CHAR" in args.model_mode:
                inputs["char_token_data"] = txt[1]
                inputs["word_token_data"] = txt[2]
                txt = txt[0]
            outputs = model(**inputs)
            # print(outputs)
            loss = outputs[0]
            # print(loss)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if type(loss) == tuple:
                # print(list(map(lambda x:x.item(),loss)))
                ep_loss.append(list(map(lambda x: x.item(), loss)))
                loss = sum(loss)
            else:
                ep_loss.append([loss.item()])

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_test_during_training:
                        results = evaluate(args, model, test_dataset, "test", global_step)
                        acc = str(results['acc'])
                    else:
                        results = evaluate(args, model, dev_dataset, "dev", global_step)
                        acc = str(results['acc'])

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-best")

                    if float(best_acc) <= float(acc):
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        torch.save(model.state_dict(), os.path.join(output_dir, "training_model.bin"))
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        with open(os.path.join(output_dir,"model_code.txt"),"w") as fp:
                            fp.writelines(inspect.getsource(MODEL_LIST[args.model_mode]))

                        logger.info("Saving model checkpoint to {}".format(output_dir))
                        temp = acc

                    if args.save_optimizer:
                        if float(best_acc) <= float(acc):
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to {}".format(output_dir))
                    best_acc = temp

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch + 1))
        mb.write("Epoch loss = {} ".format(np.mean(np.array(ep_loss), axis=0)))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ep_loss = []

    for (batch, txt) in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            if "KOSAC" in args.model_mode:
                inputs["polarity_ids"] = batch[4]
                inputs["intensity_ids"] = batch[5]
            if "KNU" in args.model_mode:
                inputs["polarity_ids"] = batch[4]
            if "CHAR" in args.model_mode:
                inputs["char_token_data"] = txt[1]
                inputs["word_token_data"] = txt[2]
                txt = txt[0]
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if type(tmp_eval_loss) == tuple:
                # print(list(map(lambda x:x.item(),tmp_eval_loss)))
                ep_loss.append(list(map(lambda x: x.item(), tmp_eval_loss)))
                tmp_eval_loss = sum(tmp_eval_loss)
            else:
                ep_loss.append([tmp_eval_loss.item()])

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if output_modes[args.task] == "classification":
        preds = np.argmax(preds, axis=1)

    result = compute_metrics(args.task, out_label_ids, preds)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))
            logger.info("Epoch loss = {} ".format(np.mean(np.array(ep_loss), axis=0)))
            f_w.write("Epoch loss = {} ".format(np.mean(np.array(ep_loss), axis=0)))

    return results


def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))
    logger.info("cliargs parameters {}".format(cli_args))


    args.output_dir = os.path.join(args.ckpt_dir, cli_args.result_dir)
    args.model_mode = cli_args.model_mode

    if os.path.exists(args.output_dir):
        raise ValueError("result path is already exist(path = %s)" % args.output_dir)

    init_logger()
    set_seed(args)

    labels = ["0", "1"]
    if output_modes[args.task] == "classification":
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
            num_labels=tasks_num_labels[args.task],
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
        )

    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    # GPU or CPU
    args.device = "cuda:{}".format(cli_args.gpu) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    config.device = args.device
    args.model_mode = cli_args.model_mode

    model = MODEL_LIST[cli_args.model_mode](args.model_type, args.model_name_or_path, config)
    model.to(args.device)
    if cli_args.small == False:
        # Load dataset
        train_dataset = DATASET_LIST[cli_args.model_mode](args, tokenizer, mode="train") if args.train_file else None
        dev_dataset = DATASET_LIST[cli_args.model_mode](args, tokenizer, mode="dev") if args.dev_file else None
        test_dataset = DATASET_LIST[cli_args.model_mode](args, tokenizer, mode="test") if args.test_file else None
    else:
        # Load dataset
        train_dataset = DATASET_LIST[cli_args.model_mode](args, tokenizer,
                                                          mode="train_small") if args.train_file else None
        dev_dataset = DATASET_LIST[cli_args.model_mode](args, tokenizer, mode="dev_small") if args.dev_file else None
        test_dataset = DATASET_LIST[cli_args.model_mode](args, tokenizer, mode="test_small") if args.test_file else None

    args.logging_steps = int(len(train_dataset) / args.train_batch_size) + 2
    args.save_steps = args.logging_steps

    if dev_dataset == None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use testset

    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset, test_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in
            sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = MODEL_FOR_SEQUENCE_CLASSIFICATION[args.model_type].from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default="koelectra-base.json")
    cli_parser.add_argument("--result_dir", type=str, required=True)
    cli_parser.add_argument("--model_mode", type=str, required=True, choices=MODEL_LIST.keys())
    cli_parser.add_argument("--gpu", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)