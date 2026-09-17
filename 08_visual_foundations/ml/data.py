"""
Project 08 -- the two real datasets behind the visual lessons.

SMS Spam Collection: 5,574 text messages labelled spam or ham. Many messages
occur more than once (the same promotion sent to many people), so a random
split can put a message in training and its exact copy in testing. Messages
are deduplicated before splitting, and the effect of not doing so is measured.

Combined Cycle Power Plant: 9,568 hourly readings of ambient temperature,
exhaust vacuum, ambient pressure and relative humidity against the plant's net
electrical output. Its features sit on very different scales -- pressure is
around 1,000, temperature around 20 -- which is exactly what makes gradient
descent without feature scaling slow or unstable.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from dsfs import Provenance, fetch, stamp_frame

SMS = Provenance(
    slug="uci-sms-spam",
    name="SMS Spam Collection (UCI)",
    source_url="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    license="Creative Commons Attribution 4.0 (CC BY 4.0)",
    citation=("Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of "
              "SMS spam filtering: New collection and results. Proceedings of DocEng 2011, 259-262."),
    notes="Tab-separated label and message.",
)

POWER = Provenance(
    slug="uci-ccpp",
    name="Combined Cycle Power Plant (UCI)",
    source_url="https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip",
    license="Creative Commons Attribution 4.0 (CC BY 4.0)",
    citation=("Tüfekci, P. (2014). Prediction of full load electrical power output of a base load operated "
              "combined cycle power plant using machine learning methods. International Journal of Electrical "
              "Power & Energy Systems, 60, 126-140."),
    notes="First sheet of Folds5x2_pp.xlsx; the other four sheets are reshuffles of the same rows.",
)

POWER_FEATURES = {"AT": "Ambient temperature (°C)", "V": "Exhaust vacuum (cm Hg)",
                  "AP": "Ambient pressure (mbar)", "RH": "Relative humidity (%)"}
POWER_TARGET = "PE"


def load_sms() -> tuple[pd.DataFrame, Provenance]:
    path, prov = fetch(SMS, filename="sms_spam_collection.zip")
    with zipfile.ZipFile(path) as z:
        raw = z.read("SMSSpamCollection").decode("utf-8", errors="replace")
    rows = [line.split("\t", 1) for line in raw.splitlines() if "\t" in line]
    df = pd.DataFrame(rows, columns=["label", "text"])
    df["spam"] = (df["label"] == "spam").astype(int)
    return df, stamp_frame(prov, df)


def load_power() -> tuple[pd.DataFrame, Provenance]:
    path, prov = fetch(POWER, filename="ccpp.zip")
    with zipfile.ZipFile(path) as z:
        df = pd.read_excel(io.BytesIO(z.read("CCPP/Folds5x2_pp.xlsx")), sheet_name=0)
    return df, stamp_frame(prov, df)


if __name__ == "__main__":
    s, _ = load_sms()
    p, _ = load_power()
    print(s.shape, s["spam"].mean(), s["text"].duplicated().sum())
    print(p.shape, p.duplicated().sum())
