# import subprocess
import os
import time
import requests
import random

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from duckduckgo_search import DDGS

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

from utils import workflow

os.environ['DISPLAY'] = ':99'

ua = UserAgent()

SERVICE_PATH = 'C:/Program Files/Google/Chrome/Application'

def BingSearchTool(query: str, read_content: bool = True, return_num: int = 5, screenshot: bool = False):
    """
    Use Selenium and BeautifulSoup to crawl Bing Search results. Optionally, read the content of each link. Use LLM to summarize the page content.

    Args:
        query (str): The search query.
        read_content (bool): Whether to read the content of each link.
    
    Returns:
        str: The search results or an error message.
    """
    url = f"https://www.bing.com/search?q={query}"
    driver = None
    headers = {'User-Agent': ua.random}
    try:
        # Initialize the WebDriver (Ensure you have a valid path for your WebDriver)
        options = Options()
        options.add_argument("--headless")  # Run in headless mode (no GUI)
        # options.add_argument("--no-sandbox")  # This is helpful in certain environments (e.g., Docker)
        # options.add_argument("--disable-dev-shm-usage")  # Helps in environments with limited shared memory
        options.add_argument(f'user-agent={headers["User-Agent"]}')

        service = Service()  # Update with the path to your chromedriver
        driver = webdriver.Chrome(service=service, options=options)

        # Open the URL and wait for the page to load
        driver.get(url)

        # Wait for the search results to be fully rendered (waiting for an element that exists in the results)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "b_algo")))

        # Get the page source after rendering the JavaScript
        html = driver.page_source
        # driver.

        # Now we can parse the rendered HTML with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        # maximum the window
        driver.maximize_window()
        
        # time.sleep(1)
        if screenshot:
            driver.save_screenshot("bing.png")

        # Extract the top 10 search results
        search_results = soup.find_all("li", class_="b_algo")
        results = []
        max_snippet_length = 1000  # Limit the snippet to 1000 characters

        for result in search_results[:return_num]:  # Get top 10 results
            title = result.find("h2").get_text()  # Extract the title
            snippet = result.find("p").get_text() if result.find("p") else "No snippet available."  # Extract the snippet (if available)

            # Truncate snippet if it's longer than the max length
            snippet = snippet[:max_snippet_length] + '...' if len(snippet) > max_snippet_length else snippet

            link = result.find("a")["href"]  # Extract the URL link from the <a> tag

            # If read_content is True, fetch and parse the content of the linked page
            if read_content:
                page_content = _fetch_page_content(link, screenshot)
                # results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}\nPage Content: {page_content}")
                results.append(page_content)
            else:
                results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")  # Just return title, snippet, and URL

        return "\n\n".join(results) if results else "No results found on Bing."
    except Exception as e:
        return f"Error during Bing search: {e}"
    finally:
        if driver:
            driver.quit()


def _fetch_page_content(url, screenshot=False):
    """Fetch and parse the content of the URL using Selenium."""

    print("Fetching page content for:", url)

    driver = None
    headers = {'User-Agent': ua.random}
    try:
        # # Start Xvfb to simulate a display if it's not already started
        # if not _is_xvfb_running():
        #     xvfb_display = subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'])
        # else:
        #     xvfb_display = None  # Xvfb is already running

        # Initialize the WebDriver (Ensure you have a valid path for your WebDriver)
        options = Options()
        options.add_argument("--headless")  # Run in headless mode (no GUI)
        # options.add_argument("--no-sandbox")  # This is helpful in certain environments (e.g., Docker)
        # options.add_argument("--disable-dev-shm-usage")  # Helps in environments with limited shared memory
        options.add_argument(f'user-agent={headers["User-Agent"]}')
        # 忽略证书错误
        options.add_argument('--ignore-certificate-errors')
        # 忽略 Bluetooth: bluetooth_adapter_winrt.cc:1075 Getting Default Adapter failed. 错误
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        # 忽略 DevTools listening on ws://127.0.0.1... 提示
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # service = Service(SERVICE_PATH)  # Update with the path to your chromedriver
        service = Service()  # Update with the path to your chromedriver

        driver = webdriver.Chrome(service=service, options=options)

        # Open the URL
        driver.get(url)

        # Wait for the page to load (waiting for a specific element, for example the body tag)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # maximum the window
        driver.maximize_window()

        # Extract the title of the page
        page_title = driver.title
        # ipdb.set_trace()

        # Extract meta description if available
        try:
            meta_description = driver.find_element(By.XPATH, "//meta[@name='description']")
            meta_description_content = meta_description.get_attribute("content") if meta_description else "No description available."
        except Exception:
            meta_description_content = "No description available."

        # # Extract all paragraphs and return all together
        # paragraphs = driver.find_elements(By.TAG_NAME, "p")
        # all_paragraph = " ".join([p.text for p in paragraphs])

        # if not all_paragraph:
            # Use JavaScript to extract all visible text from the page
        all_paragraph = driver.execute_script("return document.body.innerText;")
            

        # Quit the driver after use
        time.sleep(1)
        if screenshot:
            driver.save_screenshot(f"{url.replace('/', '_')}.png")
        driver.quit()

        # # Stop Xvfb after the task is done
        # if xvfb_display:
        #     xvfb_display.terminate()

        print("Use OpenAI to summarize the page content....")
        # Use OpenAI to summarize the page content
        page_content_summary = openai_summarize(page_title + meta_description_content + all_paragraph)
        return page_content_summary

    except Exception as e:
        if driver:
            driver.quit()
        # if xvfb_display:
        #     xvfb_display.terminate()
        return f"Error fetching page content: {e}"

GOOGLE_API = ""
SEARCH_ID = ""
def GoogleSearchTool(query: str, read_content: bool = True, search_num: int = 5, screenshot: bool = False):
    """
    Use API to get search results from Google. Optionally, fetch content from the links and summarize.
    
    Args:
        query (str): The search query.
        read_content (bool): Whether to read the content of each link.
        search_num (int): The number of search results to return.
        
    Returns:
        str: The search results or an error message.
    """
    
    try:

        # 设置你的 API 密钥和自定义搜索引擎 ID
        api_key = GOOGLE_API
        search_engine_id = SEARCH_ID

        # 构建 API 请求的 URL
        url = f'https://www.googleapis.com/customsearch/v1'

        # 定义查询参数
        params = {
            'key': api_key,  # API 密钥
            'cx': search_engine_id,  # 自定义搜索引擎 ID
            'q': query,  # 你要搜索的关键词
            'num': search_num  # 返回结果的数量
        }

        # 发送 GET 请求
        response = requests.get(url, params=params)
        search_results = []

        # 如果请求成功，解析并显示搜索结果
        if response.status_code == 200:
            results = response.json()
            for i, item in enumerate(results.get('items', []), start=1):
                title = item.get('title')
                link = item.get('link')
                snippet = item.get('snippet', 'No snippet available.')
                # print(f"Result {i}: {title}\nSnippet: {snippet}\nURL: {link}\n")
                
                # If read_content is True, fetch and parse the content of the linked page
                if read_content:
                    page_content = _fetch_page_content(link, screenshot)
                    search_results.append(page_content)
                else:
                    search_results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")
                    
            return "\n\n".join(search_results)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        return f"Error during Google search: {e}"

# @tool
def DuckDuckGoSearchTool(query: str, read_content: bool = True, return_num: int = 5):
    """
    Use the DDGS (DuckDuckGo Search) to get search results. Optionally, fetch content from the links and summarize.
    
    Args:
        query (str): The search query.
        read_content (bool): Whether to read the content of each link.
        return_num (int): The number of search results to return.
    
    Returns:
        str: The search results or an error message.
    """
    try:
        # Perform a DuckDuckGo search using DDGS
        ddgs = DDGS()
        results = ddgs.text(query, max_results = return_num)  # Modify max_results if needed

        # If no results found
        if not results:
            return "No results found on DuckDuckGo."

        search_results = []

        for result in results:
            title = result.get('title')
            snippet = result.get('snippet', 'No snippet available.')
            link = result.get('link')

            # # Simulate delay between requests to mimic human interaction
            # time.sleep(random.uniform(1.5, 3))  # Random delay between search results
            
            # If read_content is True, fetch and parse the content of the linked page
            if read_content:
                page_content = _fetch_page_content(link)
                search_results.append(page_content)
            else:
                search_results.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")

        return "\n\n".join(search_results)
    
    except Exception as e:
        return f"Error during DuckDuckGo search: {e}"
    
# use GPT to summarize the page content
def openai_summarize(text: str):
    """
    Use OpenAI's GPT-3 to summarize the input text.

    Args:
        text (str): The input text to summarize.
    
    Returns:
        str: The summarized text.
    """   
    try:
        output = workflow(
            model_name= "gpt-4o-mini",
            instruction="Assume you are a doctor, please summarize these medical-related page content into a paragraph, only keep key message, mainly focus on the symptoms, diagnosis. If this is not a medical-related page, please output 'Not a medical-related page'.",
            input_text=text[:50000]
        )
        if 'not a medical-related page' in output.lower():
            return ""
        else:
            return output
    except:
        return text[:1000]
