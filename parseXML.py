import xml.etree.ElementTree as ET
import numpy as np
import GenerateTextImages as GTI
fn = 'Data/original/a01/a01-001/strokesz.xml'
def xmltosequencepair(f):
    '''
        Input: XML file from IAM-OnDB
        Outputs:
        WrittenLines: List of List of (x, y, z) tuples corresponding to the
        strokes of a line of text. (x, y) are coordinate positions, z boolean,
        where z = 0 corresponds to the pen staying on the surface after visiting
        the (x, y) coordinate, and z = 1 corresponds to the pen leaving the
        surface (i.e. ending the stroke)
        StringLines: list of symbol images
    '''
    tree = ET.parse(f)
    root = tree.getroot()
    StringLines = getstrings(root)
    WrittenLines = roottoposnseqs(root)
    # WrittenLines = abs_to_offsets(WrittenLines)
    for i in range(len(WrittenLines)):
        WrittenLines[i] = abs_to_offsets(WrittenLines[i])


    for i in range(len(StringLines)):
        stringtoimgseq(StringLines[i])

    if len(StringLines) != len(WrittenLines):
        return None
    return (StringLines, WrittenLines)

def abs_to_offsets(WrittenLines):
    ## Converts absolute points to relative (offset) points
    new_seq = [np.zeros(3)]
    for i in range(len(WrittenLines) - 1):
        first = WrittenLines[i]
        second = WrittenLines[i + 1]
        offset = (second[0] - first[0], second[1] - first[1], second[2])
        new_seq.append(np.asarray(offset))
    return new_seq

def stringtoimgseq(strseq):
    for i in range(len(strseq)):
        char = strseq[i]
        img = GTI.symbtoimg(char)
        strseq[i] = img
    return None

def getstrings(root):
    TextLines = root.findall(".Transcription/TextLine")
    listoflines = []
    for i in range(len(TextLines)):
        text = TextLines[i].get('text')
        textlist = []
        j = 0
        while j < len(text):
            if text[j:j + 6] == '&quot;':
                textlist.append('&quot;')
                j = j + 6
            elif text[j:j + 6] == '&apos;':
                textlist.append('&apos;')
                j = j + 6
            else:
                textlist.append(text[j])
                j = j + 1
        listoflines.append(textlist.copy())
    return listoflines



def roottostrings(root):
    TextLines = root.findall(".Transcription/TextLine")
    listoflines = []
    for i in range(len(TextLines)):
        charlist = textlinetocharlist(TextLines[i])
        listoflines.append(charlist.copy())
    return listoflines

def textlinetocharlist(textline):
    Words = textline.findall("Word")
    charlist = []
    for word in Words:
        chars = word.findall('Char')
        for i in range(len(chars)):
            c = chars[i].get('text')
            if i == 0 :
                charlist.append((c, 1))
            else:
                charlist.append((c, 0))
    return charlist

def roottoposnseqs(root):
    '''
    Creates sequences of Strokes corresponding to the textlines given in the xml
    '''
    Strokes = root.findall(".StrokeSet/Stroke")
    LineStrokes = []
    for stroke in Strokes:
        points = stroke.findall("Point")
        pointlist = []
        for i in range(len(points)):
            point = points[i]
            x = int(point.get('x'))
            y = int(point.get('y'))
            if i != len(points) - 1:
                z = 0
            else:
                z = 1
            pointlist.append((x, y, z))

        if not LineStrokes:
            LineStrokes.append(pointlist.copy())
        else:
            lastStroke = LineStrokes[-1]

            lastPoint = lastStroke[-1]
            nextPoint = pointlist[0]
            if lastPoint[0] - 1500 >= nextPoint[0]:
                LineStrokes.append(pointlist.copy())
            else:
                LineStrokes[-1].extend(pointlist.copy())
    return LineStrokes
