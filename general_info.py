import requests
from bs4 import BeautifulSoup
import os
import fitz
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, ElementClickInterceptedException

def get_wiki_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    elements = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p"])
    content = []
    for elem in elements:
        tag = elem.name
        text = elem.text.strip()
        if text:
            content.append(f"{text}")
    return "\n".join(content)

def get_britannica(url):
    options = Options()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    last_height = driver.execute_script("return document.body.scrollHeight")
    no_more_loading = False
    button_xpath = "//button[contains(@class, 'js-load-next')]"
    for _ in range(70):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        try:
            load_more_button = driver.find_element(By.XPATH, button_xpath)
            driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
            time.sleep(1)
            try:
                load_more_button.click()
            except ElementClickInterceptedException:
                print("Click intercepted, using JavaScript to click.")
                driver.execute_script("arguments[0].click();", load_more_button)
            print("Clicked 'Load More' button.")
            time.sleep(1)  # Wait for more content
        except NoSuchElementException:
            print("No 'Load More' button found, continuing scrolling.")
        except StaleElementReferenceException:
                print("Stale button reference, re-locating the button.")
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            if no_more_loading:
                print("No more content to load.")
                break
            else:
                no_more_loading = True
        last_height = new_height
    content = driver.find_elements(By.TAG_NAME, "p")
    text = "\n".join([p.text for p in content])
    with open("raw_data/britannica.txt", "w") as file:
        file.write(text)
    driver.quit()

    #get the hyperlink for each form
def get_taxlinkinfo(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    elements = soup.find_all("a",class_ = "document ext-pdf")
    content = []
    for elem in elements:
        text = elem.get_text(strip=True)
        href = elem.get("href")
        content.append(f"{text}: https://www.pittsburghpa.gov/{href}")
    return "\n".join(content)

#very deep website
def deep_website(start_url, base_url, folder, name_index):
    os.makedirs(f"raw_data/{folder}", exist_ok=True)
    options = Options()
    options.add_experimental_option("prefs", {"translate": {"enabled": False}})
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    queue = [start_url]
    visited = set()
    pittsburgh_links = set()
    cached_links = set()
    colleted_link = False
    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"Visiting {url}")
        driver.get(url)
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        content_blocks = []
        for elem in soup.find_all(["p","a"]):
            if elem.name == "p":
                text = elem.get_text(strip=True)
                if text:
                    content_blocks.append(f"{text}")
            elif elem.name == "a" and colleted_link == True:
                link_text = elem.get_text(strip=True)
                href = elem.get("href")
                if href and link_text and href not in cached_links:
                    cached_links.add(href)
                    content_blocks.append(f"{link_text}: {href}")
        text_content = "\n".join(content_blocks)
        with open(f"raw_data/{folder}/{url.split('/')[name_index]}.txt", "w") as file:
            file.write(text_content)
        visited.add(url)
        all_links = [a.get("href") for a in soup.find_all("a") if a.get("href")]
        for link in all_links:
            if link.startswith(base_url) and link != start_url and link not in visited:
                pittsburgh_links.add(link)
        if not colleted_link:
            queue.extend(pittsburgh_links)
            colleted_link = True
        if len(queue) == 0:
            break
        print(f"Queue length: {len(queue)}")
        time.sleep(5)
    driver.quit()

#wiki pages
wikis_head = 'https://en.wikipedia.org/wiki/'
wikis = ['Pittsburgh','History_of_Pittsburgh']
for wiki in wikis:
    url = wikis_head + wiki
    content = get_wiki_page(url)
    #write file to txt
    with open(f"raw_data/{wiki}.txt", "w") as file:
        file.write(content)

#britannica
britannica_link = "https://www.britannica.com/place/Pittsburgh"
get_britannica(britannica_link)

#tax website
tax_web = "https://www.pittsburghpa.gov/City-Government/Finances-Budget/Taxes/Tax-Forms"
content = get_wiki_page(tax_web)
with open(f"raw_data/tax_website.txt", "w") as file:
    file.write(content)

#tax links
content = get_taxlinkinfo(tax_web)
with open(f"raw_data/tax_formlinks.txt", "w") as file:
    file.write(content)\
    
#pdf files
#pdf files
data_path = "pdf_files"

for file in os.listdir(data_path):
    if not file.endswith("pdf"): 
        continue
    file_name = os.path.join(data_path,file)
    print("Start reading file: ", file_name)
    # read doc
    doc = fitz.open(file_name)
    #extract page num
    n = doc.page_count
    doc_content = ""
    for i in range(0, n):
        page_n = doc.load_page(i)
        tables = page_n.find_tables()
        page_content = page_n.get_text("blocks")
        page_info = ""
        for element in page_content:
            if element[6] == 0:
                page_info += element[4]
            else:
                continue
        if tables.tables:
            for table in tables.tables:
                for row in table.extract():
                    page_info += "\t".join(str(cell) if cell is not None else "" for cell in row)
        doc_content += page_info + "\n"    
    txt_file = "raw_data/"+file.split("pdf")[0]+'txt'
    print("saved file: ", txt_file)
    with open(txt_file, 'w', encoding='utf-8') as file:
        file.write(doc_content)

#pittsburgh gov
deep_website("https://www.pittsburghpa.gov/Home", "https://www.pittsburghpa.gov", "citypittsburgh", -1)

#visit pittsburgh
deep_website("https://www.visitpittsburgh.com/", "https://www.visitpittsburgh.com/", "pittsburghvisit", -2)

#cmu
deep_website("https://www.cmu.edu/about/", "https://www.cmu.edu/", "cmu", -2)