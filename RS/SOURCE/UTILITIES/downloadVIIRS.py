# Check https://lpdaac.usgs.gov/data_access/data_pool is urls change
import os
import requests
from bs4 import BeautifulSoup

class SessionWithHeaderRedirection(requests.Session):
    def __init__(self, username, password):
        super().__init__()
        self.AUTH_HOST = 'urs.earthdata.nasa.gov'
        self.auth = (username, password)
   
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and redirect_parsed.hostname != self.AUTH_HOST and original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return

tile_name = 'h19v05'
yr = 2019
username = "guruprasadnk7"
password = "Monster2018"
session = SessionWithHeaderRedirection(username, password)
DATA_DIR = '../DATA/ROME/VIIRS'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# VNP09A1 : 'https://e4ftl01.cr.usgs.gov/VIIRS/VNP09A1.001/' - the 11 M bands - 1km
# VNP09H1 : 'https://e4ftl01.cr.usgs.gov/VIIRS/VNP09H1.001/' - the 3 I bands - 500m
# VNP13A1 : 'https://e4ftl01.cr.usgs.gov/VIIRS/VNP13A1.001/' - 500m vegetation indices
url_list = ['https://e4ftl01.cr.usgs.gov/VIIRS/VNP09A1.001/', 'https://e4ftl01.cr.usgs.gov/VIIRS/VNP09H1.001/']
#url_list = ['https://e4ftl01.cr.usgs.gov/MOLT/MOD09A1.006/']

for url in url_list:
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    link_list = soup.find_all('a')
    dir_list = [link.get_text() for link in link_list if link.get_text()[0:4]==str(yr)]
    file_list = []
    for dir_name in dir_list:
        new_url = url + dir_name
        print(new_url)
        page = requests.get(new_url).text
        soup = BeautifulSoup(page, 'html.parser')
        link_list = soup.find_all('a')
        file_list = file_list + [new_url+link.get_text() for link in link_list if link.get_text()[-2:] =='h5' and link.get_text().split('.')[2]==tile_name]
#        file_list = file_list + [new_url+link.get_text() for link in link_list if link.get_text()[-3:] =='hdf' and link.get_text().split('.')[2]==tile_name]
    for file in file_list:
        print(file)
        response = session.get(file, stream=True)
        with open(os.path.join(DATA_DIR, file.split("/")[-1]),'wb') as f:
            f.write(response.content)