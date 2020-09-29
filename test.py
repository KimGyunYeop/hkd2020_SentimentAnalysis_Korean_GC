import argparse
import logging
import os

import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from fastprogress.fastprogress import progress_bar
from datasets import DATASET_LIST, BaseDataset
import pandas as pd

from model import *
import json

from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    init_logger,
    compute_metrics
)

logger = logging.getLogger(__name__)


def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running Test on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running Test on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    polarity_ids = None
    intensity_ids = None
    out_label_ids = None
    txt_all = []
    ep_loss = []

    for (batch, txt) in progress_bar(eval_dataloader):
        model.eval()
        txt_all = txt_all + list(txt)
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids

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
            if "KOSAC" in args.model_mode:
                polarity_ids = inputs["polarity_ids"].detach().cpu().numpy()
                intensity_ids = inputs["intensity_ids"].detach().cpu().numpy()
            if "KNU" in args.model_mode:
                polarity_ids = inputs["polarity_ids"].detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            if "KOSAC" in args.model_mode:
                polarity_ids = np.vstack((polarity_ids, inputs["polarity_ids"].detach().cpu().numpy()))
                intensity_ids = np.vstack((intensity_ids, inputs["intensity_ids"].detach().cpu().numpy()))
            if "KNU" in args.model_mode:
                polarity_ids = np.vstack((polarity_ids, inputs["polarity_ids"].detach().cpu().numpy()))
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(out_label_ids, preds)
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

    if "KOSAC" in args.model_mode:
        return preds, out_label_ids, results, txt_all, polarity_ids, intensity_ids
    elif "KNU" in args.model_mode:
        return preds, out_label_ids, results, txt_all, polarity_ids
    else:
        return preds, out_label_ids, results, txt_all


def main(cli_args):
    # Read from config file and make args
    max_checkpoint = "checkpoint-best"

    args = torch.load(os.path.join("ckpt", cli_args.result_dir, max_checkpoint, "training_args.bin"))
    args.test_file = cli_args.test_file
    with open(os.path.join(cli_args.config_dir, cli_args.config_file)) as f:
        config = json.load(f)
        args.data_dir = config["data_dir"]
        if args.test_file == None:
            args.test_file = config["test_file"]
    logger.info("Testing parameters {}".format(args))

    args.model_mode = cli_args.model_mode
    args.device = "cuda:"+str(cli_args.gpu)

    init_logger()

    labels = ["0", "1"]
    config = CONFIG_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )

    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    args.device = "cuda:{}".format(cli_args.gpu) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    config.device = args.device
    print(args.test_file)
    # Load dataset
    test_dataset = BaseDataset(args, tokenizer, mode="test") if args.test_file else None

    logger.info("Testing model checkpoint to {}".format(max_checkpoint))
    global_step = max_checkpoint.split("-")[-1]
    model = MODEL_LIST[cli_args.model_mode](args.model_type, args.model_name_or_path, config)
    model.load_state_dict(torch.load(os.path.join("ckpt", cli_args.result_dir, max_checkpoint, "training_model.bin")))

    model.to(args.device)

    if "KOSAC" in args.model_mode:
        preds, labels, result, txt_all, polarity_ids, intensity_ids = evaluate(args, model, test_dataset, mode="test",
                                                                             global_step=global_step)
    else:
        preds, labels, result, txt_all= evaluate(args, model, test_dataset, mode="test",
                                                                               global_step=global_step)
    pred_and_labels = pd.DataFrame([])
    pred_and_labels["data"] = txt_all
    pred_and_labels["pred"] = preds
    pred_and_labels["label"] = labels
    pred_and_labels["result"] = preds == labels
    decode_result = list(
        pred_and_labels["data"].apply(lambda x: tokenizer.convert_ids_to_tokens(tokenizer(x)["input_ids"])))
    pred_and_labels["tokenizer"] = decode_result

    if "KOSAC" in args.model_mode:
        tok_an = [list(zip(x, test_dataset.convert_ids_to_polarity(y)[:len(x) + 1], test_dataset.convert_ids_to_intensity(z)[:len(x) + 1])) for x, y, z in
                  zip(decode_result, polarity_ids, intensity_ids)]
        pred_and_labels["tokenizer_analysis(token,polarity,intensitiy)"] = tok_an

    pred_and_labels.to_excel(os.path.join("ckpt", cli_args.result_dir, "test_result_" + max_checkpoint + ".xlsx"),
                             encoding="cp949")


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, default="koelectra-base.json")
    cli_parser.add_argument("--result_dir", type=str, required=True)
    cli_parser.add_argument("--model_mode", type=str, required=True, choices=MODEL_LIST.keys())
    cli_parser.add_argument("--test_file", type=str, default=None)
    cli_parser.add_argument("--gpu", type=str, default = 0)

    cli_args = cli_parser.parse_args()

    main(cli_args)
