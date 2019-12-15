from PIL import Image
from PIL import ImageDraw
import numpy as np



def symbtoimg(symb):
    ## Take a symbol as a string and converts it to an image, intended to use
    ## with the IAM-ONdb, and simple strings
    if symb.isalnum():
        img = Image.open('Alnum/{0}.png'.format(symb)).convert('L')
    elif symb == ' ':
            img = 255 * np.ones((48, 48)).astype('uint8')
    elif symb == '&apos;':
        img = Image.open('OtherSymbols/{0}.png'.format('apos')).convert('L')
    elif symb == '/':
        img = Image.open('OtherSymbols/{0}.png'.format('fslash')).convert('L')
    elif symb == '.':
        img = Image.open('OtherSymbols/{0}.png'.format('period')).convert('L')
    elif symb == '(':
        img = Image.open('OtherSymbols/{0}.png'.format('obrack')).convert('L')
    elif symb == '&quot;':
        img = Image.open('OtherSymbols/{0}.png'.format('quot')).convert('L')
    elif symb == '!':
        img = Image.open('OtherSymbols/{0}.png'.format('excl')).convert('L')
    elif symb == ')':
        img = Image.open('OtherSymbols/{0}.png'.format('cbrack')).convert('L')
    elif symb == '%':
        img = Image.open('OtherSymbols/{0}.png'.format('percent')).convert('L')
    elif symb == ',':
        img = Image.open('OtherSymbols/{0}.png'.format('comma')).convert('L')
    elif symb == '&':
        img = Image.open('OtherSymbols/{0}.png'.format('amp')).convert('L')
    elif symb == ';':
        img = Image.open('OtherSymbols/{0}.png'.format('semicolon')).convert('L')
    elif symb == '-':
        img = Image.open('OtherSymbols/{0}.png'.format('minus')).convert('L')
    elif symb == '#':
        img = Image.open('OtherSymbols/{0}.png'.format('num')).convert('L')
    elif symb == '?':
        img = Image.open('OtherSymbols/{0}.png'.format('question')).convert('L')
    elif symb == ':':
        img = Image.open('OtherSymbols/{0}.png'.format('colon')).convert('L')
    elif symb == '+':
        img = Image.open('OtherSymbols/{0}.png'.format('plus')).convert('L')
    else:
        print('Symbol $$$ {} $$$ not found'.format(symb))
        input()
        return None
    arrayout = np.asarray(img)
    arrayout = -(arrayout - 255)
    arrayout = np.reshape(arrayout, (arrayout.shape[0], arrayout.shape[1], 1)).astype('float32')
    return arrayout
