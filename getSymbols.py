import xml.etree.ElementTree as ET
import glob, os
import numpy as np
from pathlib import Path
listofpaths = []
for file_path in Path('Data/').glob('**/*.xml'):
    listofpaths.append(file_path)

allcharacters = set()
for path in listofpaths:
     tree = ET.parse(path)
     root = tree.getroot()
     allchars = root.findall('.Transcription/TextLine/Word/Char')
     for char in allchars:
         c = char.get('text')
         allcharacters.add(c)
