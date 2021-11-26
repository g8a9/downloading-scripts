import requests
import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from torchvision.transforms import functional as TF
from PIL import Image
from collections import Counter
from io import BytesIO


n_jobs = 24
headers = {
    #'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
    "User-Agent": "Googlebot-Image/1.0",  # Pretend to be googlebot
    "X-Forwarded-For": "64.18.15.200",
}


def process_row(idx, row):
    filepath = f"CC12M_scaled/CC12M_{idx:012d}.jpg"

    status = None
    if not os.path.isfile(filepath):

        try:
            # use smaller timeout to skip errors, but can result in failed downloads
            response = requests.get(
                row.url, stream=False, timeout=10, allow_redirects=True, headers=headers
            )
            status = response.status_code
        except Exception as e:
            # log errors later, set error as 408 timeout
            # print(e)
            status = 408
            return status

        if response.ok:
            try:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True

                image = Image.open(BytesIO(response.content)).convert("RGB")
                if min(image.size) > 512:
                    image = TF.resize(
                        image, size=512, interpolation=TF.InterpolationMode.LANCZOS
                    )

                # image = resize(image)  # resize PIL image
                image.save(filepath)  # save PIL image

                # with open(filepath, "wb") as out_file:
                #     out_file.write(response.content)
            except Exception as e:
                # This is if it times out during a download or decode
                # print(e)
                status = 408
                return status
        else:
            status = 408

    return status


def resize(img):
    max_size_of_short_side = 512
    if min(img.size) > max_size_of_short_side:
        img = TF.resize(img, size=max_size_of_short_side, interpolation=Image.LANCZOS)
    return img


if __name__ == "__main__":
    filename = "cc12m.tsv"
    df = pd.read_csv(filename, sep="\t", header=None, names=["url", "caption"])

    status = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(i, row)
        for i, row in tqdm(enumerate(df.itertuples()), total=len(df))
    )
    print("Images status", Counter(status))
