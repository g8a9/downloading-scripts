from os import times
import locale
from datetime import date, datetime, timedelta, timezone
import time
import requests
from tqdm import tqdm
import pandas as pd
import urllib.request
import os


locale.setlocale(locale.LC_ALL, "it_IT.UTF-8")

DAYNAMES = {
    0: "lunedi",
    1: "martedi",
    2: "mercoledi",
    3: "giovedi",
    4: "venerdi",
    5: "sabato",
    6: "domenica",
}

from bs4 import BeautifulSoup

initial_date = date(2011, 4, 23)
today = date.today()
num_days = (today - initial_date).days
print("Num days", num_days)

import re

# dd/mm/YY

results = list()
total = 0
for i in tqdm(range(num_days), desc="Days"):

    try:
        day = today - timedelta(days=i)
        day_str = day.strftime("%y%m%d")
        d1 = day.strftime("%Y/%m/%d")
        month = day.strftime("%B").lower()
        weekday = DAYNAMES[day.weekday()]
        url = f"https://www.ilpost.it/{d1}/{weekday}-{day.day}-{month}/"
        # print(url)
        req = requests.get(url)
        soup = BeautifulSoup(req.text, "lxml")

        body = soup.find_all("div", attrs={"id": "singleBody"})[0]

        def has_url(href):
            return href and href.startswith(url)

        links = body.find_all(href=has_url)

        hrefs = [l["href"] for l in links]

        for idx, href in tqdm(enumerate(hrefs), desc="imgs", leave=False):
            try:
                req = requests.get(href)
                soup = BeautifulSoup(req.text, "lxml")
                gallery = soup.find("img", attrs={"class": "photo"})
                img_url = gallery["data-src"]
                if img_url is None:
                    continue

                caption = soup.find(
                    "span", attrs={"id": "gallery-description"}
                ).text.strip()
                caption = caption.replace("\n", "")

                # filename
                ext = os.path.splitext(os.path.basename(img_url))[-1]
                filename = f"ILPOST_{day_str}_{idx:04d}{ext}"
                image_dest_path = f"ILPOST_IT/{filename}.jpg"

                if not os.path.exists(image_dest_path):
                    opener = urllib.request.build_opener()
                    opener.addheaders = [
                        ("User-Agent", "Googlebot-Image/1.0"),
                        ("X-Forwarded-For", "64.18.15.200"),
                    ]
                    urllib.request.install_opener(opener)

                    try:
                        urllib.request.urlretrieve(img_url, image_dest_path)
                    except:
                        print("Download Image as failed")
                        filename = "failed"

                else:
                    print(f"Skipping file {filename}")

                results.append(
                    {"date": day, "url": img_url, "caption": caption, "img": image_dest_path}
                )

            except Exception as e:
                print("Image at", href, "has failed")

        total += len(hrefs)

    except:
        print("Day", day, "has failed")

    if (i + 1) % 500 == 0:
        df = pd.DataFrame(results)
        df.to_csv(f"ilpost_fotodelgiorno_p{i}.tsv", sep="\t", index=None)

    time.sleep(1)

df = pd.DataFrame(results)
df.to_csv(f"ilpost_fotodelgiorno_v5.tsv", sep="\t", index=None)
