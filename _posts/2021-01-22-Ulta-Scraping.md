---
layout: post
title: "Scraping Ulta's Website for Women's Fragrance data"
subtitle: "a project on my obsession with perfume"
date: 2023-01-22
background: '/img/posts/Ulta/perfume.jpg'
---

To view my Jupyter Notebook file on my GitHub for this project, click [here](https://github.com/BritFred09/Ulta/blob/main/Ulta.ipynb). 

## Overview
For some reason, I've developed a proclivity for collecting perfumes over the past few months. I've enjoyed learning about their scent profiles, the designers' techniques, the various types of notes that exist (like hay?? pumpkin??). In an effort to practice my web scraping skills I decided to embark on a personal
project; I wanted to scrape Ulta's website for data on Women's Fragrances. After scraping the data, I wanted to do some exploratory data analysis to visualize and answer some of the following questions:

* What Fragrance Family(ies) are the most popular in Women's Fragrances? Do any particular types sell more than others?
* What Scent Types do people enjoy the most?
* What is the average price per fluid oz? Do any particular notes influence this price?
* What are the most common brands?


## Scraping Ulta's Website

![Ulta's Main Page for Women's Fragrances](/img/posts/Ulta/Ulta Main Page.png)

The first step was to scrape Ulta's Website. I knew this would be a challenge because Ulta's webpages are generated dynamically, and at the time of writing this post
they carry 329 Women's Fragrances. I wasn't going to manually go page by page for each unique product and gather data points. Instead, I utilized a Python packaged called Beautiful Soup. Beautiful Soup parses through HTML, and you can specify exactly which HTML tags you'd like to scrape via unique IDs or classes on these tags. 

## Concerns Before Starting
There were a couple of concerns I had before I started on this project, which are listed below:

* Some Perfumes had multiple size offerings. I decided to go with the size offering which was displayed by default as it was probably the most popular size sold.
* Some Perfumes did not have Scent Types, Top Notes, Middle Notes, and/or Base Notes listed. But some did. I had to check for their existence in my code prior to scraping those data points.
* Some Perfumes were refills, and not the primary product. I eliminated these products during the data cleaning process (detailed in a separate blog post).

I want to credit [this article](https://www.blog.datahut.co/post/scrape-a-dynamic-website-using-python) for helping me immensely during this project. While I didn't utilize the code provided exactly, I used the principles to complete this project. 

## Getting Started

### Step 1: import all the required packages.

```
import re
import time
import random
import warnings
import pandas as pd
from typing import List
from lxml import etree as et
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
import requests
```

### Step 2: Retrieve URLs for all page results
The perfumes were divided amongst 4 page results, so I used the following code to fetch those 4 urls. I could have done this manually, but this works too.
```
warnings.filterwarnings('ignore')
driver = webdriver.Chrome(ChromeDriverManager().install())
url = "https://www.ulta.com/shop/fragrance/womens-fragrance/perfume?N=26wq"
driver.get(url)

def get_page_urls(url):
    page_urls = [url]
    while url:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        next_page = soup.find('li', class_='next-prev floatl-span').find('a', class_='next')
        if next_page:
            url = "https://www.ulta.com" + next_page['href']
            page_urls.append(url)
        else:
            url = None
    driver.quit()
    return set(page_urls)
    

page_urls = get_page_urls(url)
page_urls
```

### Step 3: Retrive all unique URLs for each Women's Fragrance.
The next code snippet is a function that loops through all 4 page results and gets all 329 unique product URLs. 
```
def get_product_links(page_urls: List[str]) -> List[str]:
    product_links = []
    for url in page_urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        links = ["https://www.ulta.com" + row.a['href'] for row in soup.find_all('p', class_='prod-desc')]
        product_links.extend(links)
    return product_links

product_links = get_product_links(page_urls)
len(product_links)
```
### Step 4: Initiate a dataframe to hold the Perfume Data
I then initiated a pandas dataframe where the first column held those 329 unique product URLs.
```
data = {
    'Product_url': []

}

df = pd.DataFrame(data)

for i in product_links:
    df = df.append({'Product_url': i}, ignore_index=True)
```
### Step 5: Get brand names, product names, product descriptions, average ratings, and sizes. 
Next I needed to obtain the Brand Name, Product Name, Product Description, Average Rating, and Size for each product. Some products didn't have a size listed, and I used an if statement to check for this condition. If there was no size, I put "N/A" in the dataframe. I then put these datapoints in the dataframe as new columns.

```
brands = []
product_names = []
product_descriptions = []
avg_ratings = []
sizes = []

for i in range(len(product_links)):
    product = requests.get(product_links[i])
    soup = BeautifulSoup(product.text, 'lxml')
    brand = soup.find("span", attrs={"class": "Text-ds Text-ds--body-1 Text-ds--left"}).find("a").string
    product_name = soup.find("span", attrs={"class": "Text-ds Text-ds--title-5 Text-ds--left"}).string
    product_desc = soup.find("p", attrs={"class": "Text-ds Text-ds--subtitle-1 Text-ds--left"}).string
    avg_rating = soup.find("div", attrs={"class": "ReviewStars"}).find_next("span").string
    size = soup.find("span", attrs={"class": "Text-ds Text-ds--body-3 Text-ds--left Text-ds--black"})
    if size:
        size_clean = size.string
    else:
        size_clean = "N/A"
    brands.append(brand)
    product_names.append(product_name)
    product_descriptions.append(product_desc)
    avg_ratings.append(avg_rating)
    sizes.append(size_clean)

df['Brand'] = brands
df['Product Name'] = product_names
df['Product Description'] = product_descriptions
df['Size'] = sizes
```
### Step 6: Obtain Fragrance Note Info
This was the fun part for me, and the data I REALLY wanted to collect. I wanted to gather information on the notes of each perfume. 
The following data points were available on SOME perfumes for me to collect:
* Fragrance Family
* Scent Type
* Top Notes
* Middle Notes
* Bottom Notes

The annoying part: not all perfumes had each of these. Here's some screenshots to show what I mean:


![Ulta's Main Page for Women's Fragrances](/img/posts/Ulta/Ulta no Scent Details.png)
<br>
*The Chanel CHANCE Eau de Parfum Spray had absolutely no information about the Fragrance Notes.*

![Ulta's Main Page for Women's Fragrances](/img/posts/Ulta/Ulta with Scent Details.png)
<br>
*Here's an example of a Fragrance with Fragrance Family and Key Notes info, but no Scent Type*

![Ulta's Main Page for Women's Fragrances](/img/posts/Ulta/Ulta with Scent Type.png)
<br>
*Here's an example of a Fragrance with a Scent Type*

Here's my snippet of code to handle these data points. I essentially had to check for the existence of each variable before executing the necessary code to scrape the text. If an element didn't exist in the HTML, I assigned the data point as "none". Then, after collecting the data points I inserted them in to the dataframe as new columns.

```
fragrance_families_list = []
scent_types_list = []
top_notes_list = []
middle_notes_list = []
base_notes_list = []
combined_notes_list = []


for i in range(len(product_links)):
    product = requests.get(product_links[i])
    soup = BeautifulSoup(product.text, 'lxml')
    # find soup for Fragrance Info
    FragInfo = soup.find("div", attrs={"class": "Markdown Markdown--body-2"})
    frag_fam_result = FragInfo.find(string = 'Fragrance Family')
    scent_type_result = FragInfo.find(string = 'Scent Type')
    key_notes_result = FragInfo.find(string = 'Key Notes')
    # test for existence of fragrance family. If none, assign "none". if exists, assign to frgrnc_fmily_cleaned. 
    if frag_fam_result:
        frag_family = frag_fam_result.find_next('ul').find_all('li')
        frgrnc_fmly_cleaned = ''
        for j in range(len(frag_family)):
            if j < len(frag_family)-1:
                frgrnc_fmly_cleaned = frgrnc_fmly_cleaned + frag_family[j].text + ", "
            if j == len(frag_family)-1:
                frgrnc_fmly_cleaned = frgrnc_fmly_cleaned + frag_family[j].text
    else:
        frgrnc_fmly_cleaned = "none"

    # test for existence of scent type. If none, assign "none". if exists, assign to scnt_type_cleaned.
    if scent_type_result:
        scent_type = scent_type_result.find_next('ul').find_all('li')
        scnt_typ_cleaned = ''
        for j in range(len(scent_type)):
            if j < len(scent_type)-1:
                scnt_typ_cleaned = scnt_typ_cleaned + scent_type[j].text + ", "
            if j == len(scent_type)-1:
                scnt_typ_cleaned = scnt_typ_cleaned + scent_type[j].text
    else:
        scnt_typ_cleaned = "none"
        

    # test for existence of key notes. if none, assign "none" to all. If top / middle / bottom, assign respectively. if only combined, assign combined.
    if key_notes_result:
        key_notes = key_notes_result.find_next('ul').find_all('li')
        if len(key_notes) == 3:
            top_notes = key_notes[0].text
            middle_notes = key_notes[1].text
            base_notes = key_notes[2].text
            combined_notes = "none"
        if len(key_notes) == 1:
            combined_notes = key_notes[0].text
            top_notes = "none"
            middle_notes = "none"
            base_notes = "none"  
    else:
        combined_notes = "none"
        top_notes = "none"
        middle_notes = "none"
        base_notes = "none" 

    #append all variables to lists
    fragrance_families_list.append(frgrnc_fmly_cleaned)
    scent_types_list.append(scnt_typ_cleaned)
    top_notes_list.append(top_notes)
    middle_notes_list.append(middle_notes)
    base_notes_list.append(base_notes)
    combined_notes_list.append(combined_notes)

df['Fragrance Family'] = fragrance_families_list
df['Scent Type'] = scent_types_list
df['Top Notes'] = top_notes_list
df['Middle Notes'] = middle_notes_list
df['Base Notes'] = base_notes_list
df['Combined Notes'] = combined_notes_list
```

### Step 7: Obtain dynamically generated HTML content: Price and Number of Reviews.
Scraping the Price and Number of Reviews was a bit cumbersome. TLDR: in a specific script tag in the HTML (that was generated dynamically), I noted two strings: "productListPrice" and "product_reviews_count". The price and number of reviews were always listed right after these strings. So I searched for the position of these strings within the script tag, and extracted the next few characters to get the data points. You can see this in the code below. And again, I then inserted the new data points as new columns in the dataframe. 

```
prices = []
num_reviews_list = []

for i in range(len(product_links)):
    product = requests.get(product_links[i])
    soup = BeautifulSoup(product.text, 'lxml')
    script_tag = soup.find_all("script", attrs={"id": "apollo_state"})
    script_tag = str(script_tag)

    price = script_tag.partition("productListPrice")[2]
    price_find_period = price.find('.')
    price_clean = price[4:price_find_period+3]

    num_reviews = script_tag.partition("product_reviews_count")[2]
    num_reviews_first = num_reviews.find('[')+2
    num_reviews_last = num_reviews.find(']')
    num_reviews_clean = num_reviews[num_reviews_first:num_reviews_last-1]

    prices.append(price_clean)
    num_reviews_list.append(num_reviews_clean)

df['Price'] = prices
df['Number of Reviews'] = num_reviews_list
```

### COMPLETE!
That was my method for scraping Ulta's website for Women's Fragrance data. In my next post, I'll discuss how I went about cleaning this data. 