import calendar
import json
import os
import random
import re
import time
from datetime import datetime, timedelta
from turtle import title
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm


def write_json_to_txt(filename: str):
    with open(f'{filename}.json', 'r') as f:
        data = json.load(f)

    contents = []
    key_features = ['title', 'startdate', 'datetime', 'location', 'description']

    for event in data:
        optional_features = list(event.keys() - set(key_features))
        for key in key_features+optional_features:
            text = event.get(key, '')
            if not text or 'url' in key:
                continue
            if re.search(r'[\w\]\}\)]$', text):
                text += '.'
            contents.append(text)
        contents.append('')  # Add a blank line to separate events

    with open(f'{filename}.txt', 'w') as f:
        f.write('\n'.join(contents))


if __name__ == "__main__":
    home_dir = os.path.dirname(os.getcwd())
    data_dir = 'raw_data/events_pittsburgh_cmu'
    os.chdir(os.path.join(home_dir, data_dir))
    for filename in tqdm(os.listdir()):
        if filename.endswith('.json'):
            filename = os.path.splitext(filename)[0]
            write_json_to_txt(filename)
