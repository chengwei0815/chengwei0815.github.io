---
layout: post
title: "Noon Jewellery Web Scraping"
subtitle: "product and price analysis using python"
date: 2023-01-22
background: '/img/noon.png'
---

# Jewelry Data Scraping Project - Noon.com

## Overview

Noon.com is a major e-commerce platform based in the Middle East, offering a wide range of products, including electronics, fashion, beauty, home, and kitchen items. For this project, the focus is on the **jewelry category**, which offers a diverse array of items. The goal is to scrape valuable data related to jewelry products, such as product details, pricing trends, and availability, from Noon’s website. This can provide insights into current market trends, consumer preferences, and product pricing strategies.

The jewelry category on Noon contains **149 pages** with a total of **7,480 products** as of January 31, 2024. This project aims to scrape, analyze, and visualize the data from these pages.

---

## Web Scraping Process

### Tools Used:
- **Python Libraries:** `requests`, `BeautifulSoup (from bs4)`, `pandas`, `re`, and `selenium`

### Web Scraping Workflow:
We fetched and parsed product details from Noon’s jewelry section using **requests** for static content and **selenium** for dynamic content, since some data was loaded via JavaScript.

### Step 1: Setup and Library Imports


```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import random
import warnings
```

### Step 2: Retrieve URLs for All Page Results
The jewelry section consists of multiple pages, so we used selenium to automate navigation and BeautifulSoup to scrape the URLs for each page.

```python
warnings.filterwarnings('ignore')
driver = webdriver.Chrome(ChromeDriverManager().install())
url = "https://www.noon.com/uae-en/jewelry"
driver.get(url)

def get_page_urls(url):
    page_urls = [url]
    while url:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        next_page = soup.find('li', class_='next-prev floatl-span').find('a', class_='next')
        if next_page:
            url = "https://www.noon.com" + next_page['href']
            page_urls.append(url)
        else:
            url = None
    driver.quit()
    return set(page_urls)

page_urls = get_page_urls(url)
```

### Step 3: Retrieve Product Links
Once the page URLs were gathered, we extracted the unique product links for each jewelry item.

```python

def get_product_links(page_urls):
    product_links = []
    for url in page_urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        links = ["https://www.noon.com" + row.a['href'] for row in soup.find_all('p', class_='prod-desc')]
        product_links.extend(links)
    return product_links

product_links = get_product_links(page_urls)

```

### Step 4: Initialize DataFrame
We then created a pandas DataFrame to store the product details, with the initial column being the product URLs.

```python
df = pd.DataFrame({'Product_url': product_links})

```
### Step 5: Scrape Product Details
For each product, we scraped details such as the product name, brand, price, description, and other relevant information.

```python
brands = []
product_names = []
product_descriptions = []
prices = []

for link in product_links:
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'lxml')
    
    brand = soup.find("span", class_="brand").text
    name = soup.find("h1", class_="product-title").text
    description = soup.find("div", class_="product-description").text
    price = soup.find("span", class_="price").text
    
    brands.append(brand)
    product_names.append(name)
    product_descriptions.append(description)
    prices.append(price)

df['Brand'] = brands
df['Product Name'] = product_names
df['Product Description'] = product_descriptions
df['Price'] = prices


```

## Data Cleaning and Final Steps

### Cleaning:
We removed products that lacked critical information (e.g., price, size, or name).

### Analysis:
Using pandas, the data was then cleaned and analyzed to reveal insights about product prices, popular jewelry types, and customer reviews.

## Conclusion
The project provides a comprehensive dataset of jewelry products available on Noon, along with details such as product name, brand, description, fragrance information, price, ratings, and reviews. By analyzing this data, you can track pricing trends, assess popular jewelry styles, and evaluate consumer preferences.
