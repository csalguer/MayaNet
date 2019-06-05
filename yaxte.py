try:
    # For Python 3.0 and later
    from urllib.request import urlopen
    from urllib.parse import urlsplit
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
    import urlparse

from progress import ProgressBar
from os.path import basename
import sys
import os
from time import sleep
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

def simple_get(url):
    """
    Attempts to get the content at `url` by making an HTTP GET request.
    If the content-type of response is some kind of HTML/XML, return the
    text content, otherwise return None.
    """
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    """
    Returns True if the response seems to be HTML, False otherwise.
    """
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200
            and content_type is not None
            and content_type.find('html') > -1)


def log_error(e):
    """
    It is always a good idea to log errors.
    This function just prints them, but you can
    make it do anything.
    """
    print(e)


def process_FAMSI_img_tags(beauty_soup, verbose=False):
    print("Request Length: " + str(len(beauty_soup)))
    images = beauty_soup.select('img')
    print("Extracted a total of %d images" % len(images))
    src_uris= []
    glyph_lookup = {}
    for img in images[1:]:
        src = img["src"]
        title = img["title"]
        # print(src)
        title = title.lower().replace("hieroglyph", "")
        glyphname = title.strip()
        filename = basename(urlsplit(src)[2])
        if verbose:
            print("Glyph:%s ---> (%s)" % (glyphname, filename))
        glyph_lookup[filename] = glyphname
        src_uris.append(src)
    return src_uris, glyph_lookup

def download_images(uri_list, download_location, retain_original_naming=True):
    num = len(uri_list)
    progress = ProgressBar(num, fmt=ProgressBar.FULL)
    for i, src in enumerate(uri_list):
        ############################### DO WORK HERE ###########################
        try:
            img_data = urlopen(src).read()
            if len(img_data) > 0: #Read Success
                filename = basename(urlsplit(src)[2]).strip()
                if not retain_original_naming:
                    filetype = filename.split('.')[-1]
                    filename = str(i + 1) + '.' + filetype
                output = open(os.path.join(download_location, filename), 'wb')
                output.write(img_data)
                output.close()
        except Exception as e:
            log_error(e)
        ############################### END OF WORK #$##########################
        progress.current += 1
        progress()
        sleep(0.001)
    progress.done()


def process_FAMSI_html_dict(uri, dir_loc):
    print("Sending request to ", uri)
    raw_html = simple_get(uri)
    print("Request returned")

    soup = BeautifulSoup(raw_html, 'html.parser')
    src_uris, lookup = process_FAMSI_img_tags(soup)
    print('Finished compiling list of image URIs')
    print("Downloading images....")
    download_images(src_uris, dir_loc)


def prepare_FAMSI(dir='./FAMSI_dictionary_corpus/'):
    source_uri = 'http://research.famsi.org/mdp/printall.php'
    process_FAMSI_html_dict(source_uri, dir)

# prepare_FAMSI()

def pad_int_str(number):
    return "0" + str(number) if number < 10 else str(number)

def gen_dresdensis_image_uris(source_uri):
    num_pages = 74
    # Images have a naming conv for easy web scraping... len(img_label) = 21
    image_labels = ['000000' + pad_int_str(i) +'.tif.large.jpg' for i in range(1, num_pages + 1)]
    uri_list = [source_uri + img for img in image_labels]
    return uri_list


def prepare_dresdensis(dir='./codex_dresdensis'):
    source_uri = 'https://digital.slub-dresden.de/data/kitodo/codedrm_280742827/codedrm_280742827_tif/jpegs/'
    uri_list = gen_dresdensis_image_uris(source_uri)
    print('Finished compiling list of image URIs')
    print('Downlading images...')
    download_images(uri_list, dir, retain_original_naming=False)

prepare_dresdensis()
