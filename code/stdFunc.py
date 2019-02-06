# -*- coding: utf-8 -*-

import os
import json
import traceback
from collections import OrderedDict 
# from fuzzywuzzy import fuzz

# import bs4
# from bs4 import BeautifulSoup
# import urllib

import sys
reload(sys)
sys.setdefaultencoding('utf-8')



################################################################################
### pre define variables
#:: enumerate
#:: raw_input
#:: listdir
#:: sorted
### pre define function
def myjsondumps(content):
    return json.dumps(content,indent=4,ensure_ascii=False)

def mywritejson(save_path,content):
    content = json.dumps(content,indent=4,ensure_ascii=False)
    with open(save_path,'w') as f:
        f.write(content)

def myreadjson(load_path):
    with open(load_path,'r') as f:
        return json.loads(f.read())

def mywritefile(save_path,content):
    with open(save_path,'w') as f:
        f.write(content)

def myreadfile(load_path):
    with open(load_path,'r') as f:
        return f.read()

def myprint(content):
    print json.dumps(content,indent=4,ensure_ascii=False)

def rm(fi):
    os.system('rm ' + fi)

def mystrip(s):
    return ''.join(s.split())

def mysorteddict(d,key = lambda s:s):
    dordered = OrderedDict()
    for k in sorted(d.keys(),key = key):
        dordered[k] = d[k]
    return dordered

def mysorteddictfile(src,obj):
    mywritejson(obj,mysorteddict(myreadjson(src)))

def myfuzzymatch(srcs,objs,grade=80):
    matchDict = OrderedDict()
    for src in srcs:
        for obj in objs:
            value = fuzz.partial_ratio(src,obj)
            if value > grade:
                try:
                    matchDict[src].append(obj)
                except:
                    matchDict[src] = [obj]
    return matchDict

def mydumps(x):
    return json.dumps(content,indent=4,ensure_ascii=False)
