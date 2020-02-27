import sqlite3
import numpy as np
import io


# SQLITE Converter
def adapt_ndarray(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_ndarray(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)
