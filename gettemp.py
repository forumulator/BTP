import pandas as pd
import argparse
import numpy as np
from datetime import datetime
import json, re
import html
import requests

# Format according to strptime format of the python
# datetime module. See
#   https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
# for the format specification
DATETIME_FORAMT = "%d.%m.%Y %I:%M %p"

BASE_URL = """https://www.timeanddate.com/scripts/cityajax.php?n={LOC}&mode=historic&hd={HD}&month={M}&year={Y}&json=1"""
LOC = "india/guwahati"
HD = "20170226"


def find_temp_at(t, js):
    """ Find temperature from js closest to time t """
    def is_greater(dc):
        """ Check if the time in the dc dictionary is greater
            than t
        """
        def find_time(ts):
            """ Find a time string eg. 40:30 from
                the given string
            """
            print(ts)
            return re.search(r"\d{2}:\d{2}", ts).group(0)
        dt = datetime.strptime(find_time(dc["c"][0]["h"]), "%H:%M")
        return dt.time() > t
    d = next(dc for dc in js if is_greater(dc))
    temp_str = d["c"][2]["h"].split("&nbsp;")[0]
    return temp_str


def make_url(base, loc, dt):
    """ Make url for Location loc and datetime object dt for
        timeanddate.com
    """
    return base.format(LOC=loc, HD=dt.strftime("%Y%m%d"),
                       M=dt.month, Y=dt.year)


def preprocess_js_str(s):
    for c in ["c", "s", "h"]:
        s = re.sub(r'%s:' % c, r'"%s":' % c, s)
    return s


def get_temp(dt_str):
    """ Get the temperature for datetime dt """
    print(dt_str)
    dt = datetime.strptime(dt_str, DATETIME_FORAMT)
    js_str = requests.get(make_url(BASE_URL, LOC, dt)).text
    js = json.loads(preprocess_js_str(js_str))
    return int(find_temp_at(dt.time(), js))


def add_temp_to_dataset(df, col):
    """ Add column temp with the temperature entries
        to the dataset
    """
    df = df.assign(temp=np.nan)
    df["temp"] = df.apply(lambda x: (get_temp(x[col]) \
                            if type(x[col]) == str else ""), axis=1)
    return df


def parseargs():
    parser = argparse.ArgumentParser(
        description='Add temperature to RO dataset')
    parser.add_argument("-f", dest="ro_filename", required=True,
                        help="The file conataining the RO dataset")
    parser.add_argument("-c", dest="datetime_col", required=True,
                        help="The column name containing the datetime "
                             "in the filename")
    parser.add_argument("-o", dest="output_file", default=None,
                        help="Output filename")
    return parser.parse_args()


def main():
    args = parseargs()
    df = pd.read_csv(args.ro_filename)[:925]
    df = add_temp_to_dataset(df, args.datetime_col)
    if args.output_file:
        df.to_csv(args.output_file)
    else:
        print(df)


if __name__ == "__main__":
    main()
    args = parseargs()

