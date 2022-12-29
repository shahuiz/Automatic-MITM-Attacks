from io import TextIOWrapper
import gurobipy as gp
from gurobipy import GRB
from string import Template
import numpy as np
import re
import os
import math
import copy
import time
from tex_display import tex_display
from AES_solve import solve

# AES parameters
NROW = 4
NCOL = 4
NBYTE = 32
NGRID = NROW * NCOL
NBRANCH = NROW + 1
ROW = range(NROW)
COL = range(NCOL)
TAB = ' ' * 4

deployment = {
    "MulAK_switch": "NearMatch",
    "BiDir_switch": "ON",
    "GnD_switch": "OFF",
    "RKc_switch": "OFF",
    "E_OBJ": 1
}

solve(
    key_size=192, 
    total_round=10, 
    enc_start_round=3, 
    match_round=8, 
    key_start_round=3, 
    control_panel=deployment,
    dir='./AES192_10r_search/runs/'
)

