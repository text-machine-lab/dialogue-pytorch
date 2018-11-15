import urllib.request
from bs4 import BeautifulSoup
import re
import sys

assert sys.version_info[0] == 3

def remove_tags(s, tags=['a', 'p']):
    """Remove tags but keep their content
       Perform it in pairs to avoid removing special characters
    """
    for tag in tags:
        groups = re.findall('</?%s.*?>' % tag, s)
        for group in groups:
            s = s.replace(group, '')
    return s


def get_paragraphs(url, timeout=10):
    try:
        page = urllib.request.urlopen(url, timeout=timeout)

        soup = BeautifulSoup(page, 'html.parser')
    except:
        return None

    if hasattr(soup, 'title') and hasattr(soup.title, 'text'):
        title = soup.title.text
    else:
        title = ''

    if hasattr(soup, 'p'):
        paragraphs = soup.find_all('p')
        paragraphs = [remove_tags(p.text) for p in paragraphs]
    else:
        paragraphs = []

    return {'title': title, 'paragraphs': paragraphs}

if __name__ == '__main__':
    # example
    url = 'http://www.uml.edu'
    text = get_paragraphs(url)

    if text is not None:
        print(text['title'])
        print([' '.join(p.split()) for p in text['paragraphs']])