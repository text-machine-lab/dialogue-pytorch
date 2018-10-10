import yaml
import sys
import bz2
from get_url import get_paragraphs
import argparse

parser = argparse.ArgumentParser(description='Extract Reddit conversations')
parser.add_argument('--source', default=None, help='Reddit file to extract from')
parser.add_argument('--dest', default=None, help='Where to store processed tsv file')
parser.add_argument('--subreddit', default=None, help='Filter conversations to a specific subreddit')

args = parser.parse_args()

if args.source is None:
    print('Must specify source file!')
    exit()

yamlfile = bz2.BZ2File(open(args.source, 'rb'))

if args.dest is not None:
    out_file = open(args.dest, 'w')
else:
    out_file = None

temp = []
fixedlines = []
for line in yamlfile:
    line = line.decode()
    if line != "{}\n":
        if line[0].isdigit():
            if len(temp) > 3:
                joinedlines = "".join(temp)
                loadedyaml = next(iter(yaml.load(joinedlines).values()))

                if args.subreddit is None or args.subreddit == loadedyaml['subreddit']:
                    print('Reading yaml')
                    url = loadedyaml['url']
                    #print(url)
                    url_data = None
                    if not url.endswith('jpg') or not url.endswith('png'):
                        url_data = get_paragraphs(url)
                    else:
                        print('Is image')

                    if url_data is not None and not 'http' in loadedyaml['conv']:
                        print('Writing')
                        conv = [' '.join(p.split()).split(': ')[-1] for p in loadedyaml['conv'].split('\n')][:-1]
                        num_turns = len(conv)
                        conv_title = ' '.join(loadedyaml['title'].split())
                        url_title = ' '.join(url_data['title'].split())
                        url_paragraphs = ' '.join([' '.join(p.split()) for p in url_data['paragraphs']])
                        fixedlines.append([conv_title, conv, url_title, url_paragraphs])
                        if out_file is not None:
                            out_file.write(conv_title + '\t' + str(num_turns) + '\t' + '\t'.join(conv) + '\t'
                                           + url_title + '\t' + url_paragraphs + '\n')
                    else:
                        print('Broken link')
            temp = []
        temp.append(line)

if out_file is None:
    print('Printing examples...')
    print('Number of examples: %s' % len(fixedlines))
    for i in range(5):
        print(fixedlines[i][0])
        print(fixedlines[i][1])
        print()
        print(fixedlines[i][2])
        print(fixedlines[i][3])
        print()