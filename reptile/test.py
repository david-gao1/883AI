import requests
from bs4 import BeautifulSoup

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0;Win64;x64) AppleWebKit/537.36 (KHTML, like '
                         'Gecko) Chrome/85.0.4183.83 Safari/537.36'}

URL = "https://sanya.xiaozhu.com/"

res = requests.get(URL, headers=headers)  # 方法加入请求头

soup = BeautifulSoup(res.text, 'html.parser')
print(soup.prettify())
