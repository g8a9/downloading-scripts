from pandas.io import pickle
from datasets import Dataset
from transformers import MarianTokenizer, MarianMTModel
import torch
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from easynmt import EasyNMT
import pickle


def main():
    p = ArgumentParser()
    p.add_argument("--beam_size", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--save_every_n_batch", type=int, default=500)
    args = p.parse_args()

    print("Loading dataset")
    df = pd.read_csv("cc12m.tsv", sep="\t", header=None, names=["url", "caption"])
    dataset = Dataset.from_pandas(df)

    model_name = "Helsinki-NLP/opus-mt-en-it"
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    def transform(examples):
        return tokenizer(
            examples["caption"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )

    dataset.set_transform(transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    translations = list()
    checkpoint = list()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            translated = model.generate(
                **batch,
                num_beams=args.beam_size,
                early_stopping=True,
                max_length=args.max_length,
            )
        translated = [t.cpu() for t in translated]
        output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        translations.extend(output)
        checkpoint.extend(output)

        if batch_idx % args.save_every_n_batch == 0:
            ckpt = pd.Series(checkpoint)
            ckpt.to_csv(
                f"CC13M_checkpoint_{batch_idx}.gz",
                index=None,
                header=None,
                compression="gzip",
            )
            checkpoint = list()

    # model = EasyNMT(
    #     "opus-mt",  #  "m2m_100_1.2B",  #  "opus-mt"
    #     device=0,
    #     max_length=args.max_length,
    #     cache_folder="easynmt_cache",
    # )

    # print("run_translations")
    # translations = model.translate(
    #     df.caption,
    #     source_lang="en",
    #     target_lang="it",
    #     batch_size=args.batch_size,
    #     beam_size=args.beam_size,
    #     show_progress_bar=True,
    # )

    df["caption_it"] = translations
    df.to_csv("cc12m_translated.tsv", sep="\t", index=None)


if __name__ == "__main__":
    main()
