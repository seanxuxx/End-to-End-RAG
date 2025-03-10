import calendar
import json
import os
import random
import re
import time
from datetime import datetime, timedelta
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

STARTING_DATE = datetime(2025, 3, 19)
ENDING_DATE = datetime(2025, 12, 31)

PITTSBURGH_EVENTS_URL = 'https://pittsburgh.events/'
DOWNTOWN_PITTSBURGH_URL = 'https://downtownpittsburgh.com/events/'
PGH_CITY_PAPER_URL = 'https://www.pghcitypaper.com/pittsburgh/EventSearch'
CMU_EVENT_URL = 'https://events.cmu.edu/day/date/'
CMU_COMMUNITY_URL = 'https://community.cmu.edu/s/events'


# ============================================================================ #
# Save file functions
# ============================================================================ #


def save_file(events: list[dict], filename: str, save_json: bool, save_txt: bool):
    if save_json:
        save_events_to_json(filename, events)
    if save_txt:
        write_json_to_txt(filename)


def save_events_to_json(filename: str, events: list[dict]):
    with open(f'{filename}.json', 'w') as f:
        json.dump(events, f, indent=4)
    print(f'Saved scraped data to {filename}.json')


def write_json_to_txt(filename: str):
    json_file = f'{filename}.json'
    if not os.path.exists(json_file):
        return

    with open(json_file, 'r') as f:
        data = json.load(f)

    contents = []
    key_features = ['title', 'startdate', 'datetime', 'location', 'description']

    for event in data:
        for key in ['url', 'date', 'time']:
            event.pop(key, None)
        # Add a period at the end of the text if it doesn't end with a punctuation
        for key, text in event.items():
            if re.search(r'[\w\]\}\)]$', text):
                event[key] = f'{text}.'
        # Format the text
        optional_features = sorted(list(event.keys() - set(key_features)))
        basic_info = ' '.join([event[key] for key in key_features[:-1]
                              if key in event])
        optional_info = ' '.join([event[key] for key in optional_features])
        formatted_text = [basic_info, optional_info]
        if 'description' in event:
            formatted_text.insert(1, event['description'])
        # Join all text into one line
        contents.append(' '.join(formatted_text))
        # Add a blank line to separate events
        contents.append('')

    txt_dir = os.path.join(os.path.dirname(os.getcwd()), 'events_formatted')
    os.makedirs(txt_dir, exist_ok=True)
    txt_filepath = os.path.join(txt_dir, f'{filename}.txt')
    with open(txt_filepath, 'w') as f:
        f.write('\n'.join(contents))
    print(f'Write data to {txt_filepath}')


def convert_all_jsons():
    for filename in tqdm(os.listdir()):
        if filename.endswith('.json'):
            filename = os.path.splitext(filename)[0]
            write_json_to_txt(filename)


# ============================================================================ #
# Pittsburgh Events
# ============================================================================ #


def load_pgh_event_by_month(month: str):
    """
    Load all the events in this month on pittsburgh.events

    Args:
        month (str): full month name in lowercase
    """
    driver.get(urljoin(PITTSBURGH_EVENTS_URL, month))
    while True:
        try:
            show_more_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'ldm'))
            )
            show_more_button.click()
            time.sleep(5)
        except:
            break


def scrape_pgh_event_by_month(month: str) -> list[dict]:
    """
    Scrape event info from a month-specific page of pittsburgh.events

    Args:
        month (str): full month name in lowercase

    Returns:
        list[dict]: A list of scraped event dictionaries
    """
    events = []
    try:
        event_ul = driver.find_element(By.XPATH, '//ul[contains(@class, "dates-list")]')
        # Extract event info from each <li> tag
        for event_li in tqdm(event_ul.find_elements(By.TAG_NAME, 'li'),
                             desc=f'Scraping {month}'):
            event = {}
            find_mapping = {
                'url': (By.TAG_NAME, 'a'),
                'title': (By.TAG_NAME, 'a'),
                'date': (By.CLASS_NAME, 'date'),
                'time': (By.CLASS_NAME, 'time'),
                'datedesc': (By.CLASS_NAME, 'date-desc'),
                'location': (By.CLASS_NAME, 'location'),
                'price': (By.CLASS_NAME, 'from-price'),
            }
            for key, (by, value) in find_mapping.items():
                try:
                    element = event_li.find_element(by, value)
                    if key == 'url':
                        text = element.get_attribute('href')
                    else:
                        text = element.text
                    text = text.replace('\n', ' ').strip()
                    event[key] = text
                except:
                    pass
            if 'date' in event and 'time' in event:  # Format datetime
                event['datetime'] = f'{event['date']}, {event['time']}'
            if len(event) > 0:  # Valid event info
                events.append(event)
    except:
        print(f'{driver.current_url}: No event list found.')
    time.sleep(random.uniform(2, 10))
    return events


def get_pittsburgh_events(filename: str, save_json=True, save_txt=True):
    events = []
    print(f'Scraping from {PITTSBURGH_EVENTS_URL}')

    # Scrape events by month
    for month in calendar.month_name[3:]:
        month = month.lower()
        load_pgh_event_by_month(month)
        month_events= scrape_pgh_event_by_month(month)
        events.extend(month_events)
    print(f'Scraped {len(events)} events')

    save_file(events, filename, save_json, save_txt)


# ============================================================================ #
# Downtown Pittsburgh Events
# ============================================================================ #


def scrape_pdp_event_item(event_item: Tag) -> dict:
    """
    Args:
        event_item (bs4.element.Tag): eventitem div tag from Downtown Pittsburgh event page

    Returns:
        dict
    """
    # Extract basic information from eventitem
    result = {
        'url': urljoin(DOWNTOWN_PITTSBURGH_URL, event_item.a.attrs['href']),
        'title': event_item.a.get_text(strip=True),
        'datetime': re.sub(r'\s{1,}', ' ',
                           event_item.find(class_='eventdate').get_text(' ', strip=True).replace('|', ',')),
    }

    # Load the event page and extract event details from READ MORE
    response = requests.get(result['url'])
    assert response.status_code == 200, f'Failed to fetch {response.url}: {response.status_code}'
    soup = BeautifulSoup(response.content, 'html.parser')
    description = []
    for element in soup.find('div', class_='eventitem').find('div', class_='copyContent').contents:  # type: ignore
        text = element.get_text(' ', strip=True)
        if len(text) == 0 or element.name == 'h1':  # Skip empty text or title
            continue
        elif 'class' in element.attrs:
            if element.find('a'):  # Extract hyperlink instead of plain text
                text += f": {element.find('a').get('href')}"
            result[element.attrs['class'][0]] = text
        else:  # Add all other text to description
            if not text.endswith('.'):
                text += '.'
            description.append(text)
    result['description'] = ' '.join(description)
    result['location'] = result.pop('eventlocation', '')  # Rename location key
    result.pop('eventdate', None)  # Remove redundant datetime
    return result


def get_pdp_events(filename: str, save_json=True, save_txt=True):
    print(f'Scraping from {DOWNTOWN_PITTSBURGH_URL}')
    events = []
    response = requests.get(DOWNTOWN_PITTSBURGH_URL)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        for event_item in tqdm(soup.find_all('div', class_='eventitem'),
                               desc='Scraping events'):
            curr_event = scrape_pdp_event_item(event_item)
            events.append(curr_event)
        print(f'Scraped {len(events)} events')
        save_file(events, filename, save_json, save_txt)
    else:
        print(f'Failed to fetch {response.url}: {response.status_code}')


# ============================================================================ #
# Pittsburgh City Paper Events
# ============================================================================ #


def scrape_pghcitypaper_search_page(url: str) -> list[dict]:
    """
    Args:
        url (str): Pittsburgh City Paper event calendar page URL,
        e.g. https://www.pghcitypaper.com/pittsburgh/EventSearch?v=d

    Returns:
        list[dict]: A list of scraped event dictionaries
    """
    events = []
    driver.get(url)
    try:
        event_ul = driver.find_element(By.CLASS_NAME, 'search-results')
        # Extract event info from each <li> tag
        for event_li in tqdm(event_ul.find_elements(By.TAG_NAME, 'li'),
                             desc=f'Scraping {urlparse(url).query}'):
            event = {}
            find_mapping = {
                'url': 'fdn-teaser-headline',
                'title': 'fdn-teaser-headline',
                'datetime': 'fdn-teaser-subheadline',
                'location': 'fdn-event-teaser-location-block',
                'category': 'fdn-teaser-tag-link',
                'price': 'fdn-pres-details-split',
                'ticketlink': 'fdn-teaser-ticket-link',
                'description': 'fdn-teaser-description',
            }  # Value: class name
            for key, class_name in find_mapping.items():
                try:
                    element = event_li.find_element(By.CLASS_NAME, class_name)
                    if key == 'url':
                        text = element.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    elif key == 'ticketlink':
                        text = f"Get ticket: {element.get_attribute('href')}"
                    else:
                        text = element.text.strip()
                        text = re.sub(r'\s{1,}', ' ', text)  # Remove extra spaces
                    event[key] = text
                except:
                    pass
            if len(event) > 0:
                events.append(event)
    except:
        print(f'{url}: No event list found.')
    time.sleep(random.uniform(2, 10))
    return events


def get_pghcitypaper_events(filename: str, save_json=True, save_txt=True):
    events = []
    driver.get(PGH_CITY_PAPER_URL)
    print(f'Scraping from {PGH_CITY_PAPER_URL}')

    # Scape page by page
    while True:
        events.extend(scrape_pghcitypaper_search_page(driver.current_url))
        try:  # Turn to next page
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH,
                     '//a[contains(@class, "fdn-page-navigation-prev-next") and contains(text(), "next")]'))
            )
            next_button.click()
        except:
            print('No next page found.')
            break
    print(f'Scraped {len(events)} events')

    save_file(events, filename, save_json, save_txt)


# ============================================================================ #
# CMU Events
# ============================================================================ #


def scrape_cmu_event_calendar(url: str) -> list[str]:
    """
    Args:
        url (str): CMU Events Calendar by day page URL,
        e.g. https://events.cmu.edu/day/date/20250319

    Returns:
        list[str]: A list of event URLs
    """
    event_urls = []
    driver.get(url)
    try:  # Find the event list container
        event_container = driver.find_element(By.CLASS_NAME, 'lw_cal_event_list')
        # Select all child elements
        for event_item in event_container.find_elements(By.XPATH, './*'):
            try:
                event_url_tag = (event_item
                                 .find_element(By.CLASS_NAME, 'lw_events_title')
                                 .find_element(By.TAG_NAME, 'a'))
                event_url = event_url_tag.get_attribute('href')
                event_urls.append(event_url)
            except:
                pass
    except:
        print(f'{url}: No event list found.')
    time.sleep(random.uniform(2, 10))
    return event_urls


def scrape_cmu_event_page(url: str) -> dict:
    """
    Args:
        url (str): CMU Event Calendar event page URL, e.g. https://events.cmu.edu/event/12496-ramadan

    Returns:
        dict
    """
    result = {'url': url}
    driver.get(url)
    contents = driver.find_elements(By.ID, 'main-content')
    if len(contents) > 0:  # Check if main content is found
        content = contents[0]
        find_mapping = {
            'title': (By.TAG_NAME, 'h1'),
            'startdate': (By.ID, 'lw_cal_this_day'),
            'datetime': (By.XPATH, '//h1/following-sibling::p'),
            'contact': (By.ID, 'lw_cal_event_leftcol'),
            'description': (By.ID, 'lw_cal_event_rightcol'),
        }
        for key, (by, value) in find_mapping.items():
            try:
                text = content.find_element(by, value).text.strip()
                text = re.sub(r'\s{1,}', ' ', text)  # Remove extra spaces
                result[key] = text
            except:
                pass
    time.sleep(random.uniform(2, 10))
    return result


def scrape_timely_event_page(url: str) -> dict:
    """
    Args:
        url (str): Timely event page URL, e.g. https://events.time.ly/vdibqnd/43900770

    Returns:
        dict
    """
    result = {'url': url}
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        result['title'] = soup.h1.get_text(' ', strip=True)
        # Scrape details
        container = soup.find(class_='timely-event-details')
        if container:
            for detail in container.find_all(class_='timely-details'):
                key, value = [tag.get_text(' ', strip=True) for tag in detail.contents
                              if len(tag.get_text(' ', strip=True)) > 0]
                result[key] = value
        # Rename detail keys
        key_mapping = {
            'WHEN': 'datetime',
            'WHERE': 'location',
            'COST': 'price',
            'CONTACT': 'contact',
        }
        for old_key, new_key in key_mapping.items():
            if old_key in result:
                result[new_key] = result.pop(old_key)
        # Scrape description
        description = soup.find(class_='timely-event-has-description')
        if description:
            text = description.get_text(' ', strip=True)
            text = re.sub(r'\s{1,}', ' ', text)  # Remove extra spaces
            result['description'] = text
    else:
        print(f'Failed to fetch {url}: {response.status_code}')
    return result


def get_cmu_events(filename: str, save_json=True, save_txt=True,
                   start_date=STARTING_DATE, end_date=ENDING_DATE):
    # Get all the event URLs from the Event Calendar
    event_urls = []
    days = (end_date - start_date).days
    print(f'Scraping from {CMU_EVENT_URL}')
    for d in tqdm(range(days + 1), desc='Scraping CMU calendar'):
        curr_date = start_date + timedelta(days=d)
        url = CMU_EVENT_URL + curr_date.strftime('%Y%m%d')
        event_urls.extend(scrape_cmu_event_calendar(url))

    # Scrape each event page
    events = []
    for url in tqdm(set(event_urls), desc='Scraping events'):
        domain = urlparse(url).netloc
        if domain == 'events.cmu.edu':
            curr_event = scrape_cmu_event_page(url)
        else:
            curr_event = scrape_timely_event_page(url)
        events.append(curr_event)
    print(f'Scraped {len(events)} events')

    save_file(events, filename, save_json, save_txt)

# ============================================================================ #
# Campus Events
# ============================================================================ #


def scrape_cmu_community_search_grid(url='') -> list[str]:
    """
    Args:
        url (str, optional): Defaults to ''.

    Returns:
        list[str]: A list of event URLs from CMU Alumni Community event
    """
    if url != '':
        driver.get(url)

    # Get the page number
    try:
        page = driver.find_element(By.CLASS_NAME, 'slds-text-body_small').text
    except:
        page = 'no page number'

    # Extract all the event urls in the grid
    event_urls = []
    try:
        event_container = driver.find_element(By.CSS_SELECTOR, '.slds-grid.slds-wrap.cCMU_Theme')
        # Extract event url from each <div> tag
        for event_div in tqdm(event_container.find_elements(By.XPATH, './*'),
                              desc=f'Scraping {page}'):
            try:
                event_tag = event_div.find_element(By.CLASS_NAME, 'evt_name')
                event_url = event_tag.get_attribute('href')
                event_urls.append(event_url)
            except:
                pass
    except:
        print(f'{driver.current_url}: No event list found.')
    time.sleep(random.uniform(2, 10))
    return event_urls


def scrape_givecampus_event_page(url: str) -> dict:
    """
    Args:
        url (str): Givecampus event page URL,
        e.g. https://www.givecampus.com/schools/CarnegieMellonUniversity/events/phl-women-s-history-month-dinner-in-philly

    Returns:
        dict
    """
    result = {'url': url}
    temp_driver = webdriver.Chrome()  # Reinitialize to pass human verification
    temp_driver.get(url)
    contents = temp_driver.find_elements(By.ID, 'main-content')
    if len(contents) > 0:  # Check if main content is found
        content = contents[0]
        find_mapping = {
            'title': [(By.TAG_NAME, 'h1')],
            'datetime': [(By.ID, 'event-when'), (By.CLASS_NAME, 'text-left')],
            'location': [(By.ID, 'event-where'), (By.CLASS_NAME, 'text-left')],
            'price': [(By.ID, 'event-price'), (By.CLASS_NAME, 'text-left')],
            'message': [(By.ID, 'event-purchase-message')],
            'description': [(By.ID, 'event-description')],
        }
        for key, by_value_pairs in find_mapping.items():
            try:
                element = content
                for (by, value) in by_value_pairs:
                    element = element.find_element(by, value)
                text = element.text.strip()
                if key == 'datetime' or key == 'location':
                    text = text.replace('\n', ', ')
                text = re.sub(r'\s{1,}', ' ', text)  # Remove extra spaces
                result[key] = text
            except:
                pass
    time.sleep(random.uniform(2, 10))
    temp_driver.quit()
    return result


def get_cmu_community_events(filename: str, save_json=True, save_txt=True):
    # Get all the event URLs from the Event Calendar
    event_urls = []
    driver.get(CMU_COMMUNITY_URL)
    print(f'Scraping from {CMU_COMMUNITY_URL}')

    # Scape page by page
    next_button_enabled = True
    while next_button_enabled:
        event_urls.extend(scrape_cmu_community_search_grid())
        # Check next button status
        for button in driver.find_elements(By.TAG_NAME, 'button'):
            if button.text == 'Next':
                if button.get_attribute('disabled'):
                    next_button_enabled = False
        try:  # Turn to next page
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'slds-p-left--xx-small'))
            )
            next_button.click()
        except:
            pass

    # Scrape each event page
    events = []
    for url in tqdm(set(event_urls), desc='Scraping events'):
        curr_event = scrape_givecampus_event_page(url)
        events.append(curr_event)
    print(f'Scraped {len(events)} events')

    save_file(events, filename, save_json, save_txt)


if __name__ == '__main__':
    # * Check working directory
    home_dir = os.path.dirname(os.getcwd())
    data_dir = os.path.join(home_dir, 'raw_data/events_pittsburgh_cmu')
    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)

    save_json = True
    save_txt = True
    web_scraping = False

    if web_scraping:
        driver = webdriver.Chrome()
        driver.implicitly_wait(2)

        get_pittsburgh_events('pittsburgh_events', save_json, save_txt)
        get_pdp_events('downtown_pittsburgh_events', save_json, save_txt)
        get_pghcitypaper_events('pittsburgh_city_paper_events')
        get_cmu_events('cmu_events', save_json, save_txt,
                       start_date=STARTING_DATE,
                       end_date=ENDING_DATE)
        get_cmu_community_events('cmu_community_events', save_json, save_txt)

        driver.quit()

    else:
        convert_all_jsons()
