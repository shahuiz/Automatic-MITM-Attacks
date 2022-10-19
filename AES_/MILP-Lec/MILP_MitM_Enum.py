#!/usr/bin/env python
# coding: utf-8

# Mixed-Integer Linear Programming Use Case: 
# Automatic Search of Meet-in-the-Middle Preimage Attacks on AES-like Hashing
# Simplified version: without ARK

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import sys
from os import system

NRow = 4
NCol = 4
NMat = NRow * NCol
N_Br = NRow + 1
N_IO = 2 * NRow
WLastMC = 0 # WLastMC = 1 if the last round has MixColumns 

# Declare and initialize model

class MitM_Attack_Searcher():
    def __init__(self,
        aTR=3,
        aini_r=0,
        amat_r=2,
        afpath='./'):
        self.TR = aTR
        self.ini_r = aini_r # ini_r in {0,1,2,...,TR-1}
        self.mat_r = amat_r # mat_r in {0,1,2,...,TR-1}, ini_r != mat_r
        self.F_r = []
        self.B_r = []
        if self.ini_r < self.mat_r:
            self.F_r = list(range(self.ini_r, self.mat_r))
            self.B_r = list(range(self.mat_r + 1, self.TR)) + list(range(0, self.ini_r))
        else: # self.mat_r < self.ini_r
            self.B_r = list(range(self.mat_r + 1, self.ini_r))
            self.F_r = list(range(self.ini_r, self.TR)) + list(range(0, self.mat_r))
        self.fpath = afpath
        self.modelName = self.fpath + '/model_%d_x_%d_%dR_int_r%d_mat_r%d' % (NRow, NCol, self.TR, self.ini_r, self.mat_r)
        self.m = gp.Model(self.modelName)
        self.SB_x = np.ndarray(shape=(self.TR, NRow, NCol), dtype='object')
        self.SB_y = np.ndarray(shape=(self.TR, NRow, NCol), dtype='object')
        self.SB_g = np.ndarray(shape=(self.TR, NRow, NCol), dtype='object')
        self.SB_w = np.ndarray(shape=(self.TR, NRow, NCol), dtype='object')
        self.MC_x = np.ndarray(shape=(self.TR, NRow, NCol), dtype='object')
        self.MC_y = np.ndarray(shape=(self.TR, NRow, NCol), dtype='object')
        self.MC_g = np.ndarray(shape=(self.TR, NRow, NCol), dtype='object')
        self.MC_w = np.ndarray(shape=(self.TR, NRow, NCol), dtype='object')
        self.SB_X = np.ndarray(shape=(self.TR, NCol), dtype='object')
        self.SB_Y = np.ndarray(shape=(self.TR, NCol), dtype='object')
        self.SB_W = np.ndarray(shape=(self.TR, NCol), dtype='object')
        self.MC_X = np.ndarray(shape=(self.TR, NCol), dtype='object')
        self.MC_Y = np.ndarray(shape=(self.TR, NCol), dtype='object')
        self.MC_W = np.ndarray(shape=(self.TR, NCol), dtype='object')
        self.DoF_init_BL = np.ndarray(shape=(NRow, NCol), dtype='object')
        self.DoF_init_RD = np.ndarray(shape=(NRow, NCol), dtype='object')
        self.CD_x = np.ndarray(shape=(self.TR, NCol), dtype='object')
        self.CD_y = np.ndarray(shape=(self.TR, NCol), dtype='object')
        self.MT_m = np.empty(shape=(NCol), dtype='object')
        self.MT_n = np.empty(shape=(NCol), dtype='object')
        self.MT_w = np.ndarray(shape=(NRow, NCol), dtype='object')
        self.DoF_BL = None
        self.DoF_RD = None
        self.DoM = None
        self.Obj = None

    def printStateVarsName(self, aVars):
        nstr = ''
        for i in range(NRow):
            for j in range(NCol):
                nstr += aVars[i,j].VarName + ', '
            nstr += "\n"
        nstr += "\n"
        print(nstr)

    def addVars(self):
        # SB
        for ri in range(self.TR):
            for i in range(NRow):
                for j in range(NCol):
                    self.SB_x[ri, i, j] = self.m.addVar(vtype=GRB.BINARY, name="SB_x" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri))
                    self.SB_y[ri, i, j] = self.m.addVar(vtype=GRB.BINARY, name="SB_y" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri))
                    self.SB_g[ri, i, j] = self.m.addVar(vtype=GRB.BINARY, name="SB_g" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri))
                    self.SB_w[ri, i, j] = self.m.addVar(vtype=GRB.BINARY, name="SB_w" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri))
        # MC and SB point to the same variables, since SR operation simply permutate the cells
        for ri in range(self.TR):
            for i in range(NRow):
                for j in range(NCol):
                    self.MC_x[ri, i, j] = self.SB_x[ri, i, (j + i)%NCol]
                    self.MC_y[ri, i, j] = self.SB_y[ri, i, (j + i)%NCol]
                    self.MC_g[ri, i, j] = self.SB_g[ri, i, (j + i)%NCol]
                    self.MC_w[ri, i, j] = self.SB_w[ri, i, (j + i)%NCol]
        for ri in range(self.TR):
            for j in range(NCol):
                self.SB_X[ri, j] = self.m.addVar(vtype=GRB.BINARY, name="SB_Xc" + ('[%d]' % j) + "_r" + ('[%d]' % ri))
                self.SB_Y[ri, j] = self.m.addVar(vtype=GRB.BINARY, name="SB_Yc" + ('[%d]' % j) + "_r" + ('[%d]' % ri))
                self.SB_W[ri, j] = self.m.addVar(vtype=GRB.BINARY, name="SB_Wc" + ('[%d]' % j) + "_r" + ('[%d]' % ri))
                self.MC_X[ri, j] = self.m.addVar(vtype=GRB.BINARY, name="MC_Xc" + ('[%d]' % j) + "_r" + ('[%d]' % ri))
                self.MC_Y[ri, j] = self.m.addVar(vtype=GRB.BINARY, name="MC_Yc" + ('[%d]' % j) + "_r" + ('[%d]' % ri))
                self.MC_W[ri, j] = self.m.addVar(vtype=GRB.BINARY, name="MC_Wc" + ('[%d]' % j) + "_r" + ('[%d]' % ri))
        # DoF_init
        for i in range(NRow):
            for j in range(NCol):
                self.DoF_init_BL[i, j] = self.m.addVar(vtype=GRB.BINARY, name="DoF_init_BL" + ('[%d,%d]' % (i,j)))
                self.DoF_init_RD[i, j] = self.m.addVar(vtype=GRB.BINARY, name="DoF_init_RD" + ('[%d,%d]' % (i,j)))
        for ri in range(self.TR):
            for j in range(NCol):
                self.CD_x[ri, j] = self.m.addVar(lb=0, ub=NRow, vtype=GRB.INTEGER, name="CD_x_c" + ('[%d]' % j) + "_r" + ('[%d]' % ri))
                self.CD_y[ri, j] = self.m.addVar(lb=0, ub=NRow, vtype=GRB.INTEGER, name="CD_y_c" + ('[%d]' % j) + "_r" + ('[%d]' % ri))
        for j in range(NCol):
            self.MT_n[j] = self.m.addVar(lb=-NRow, ub=NRow, vtype=GRB.INTEGER, name="MT_n_c" + ('[%d]' % j))
            self.MT_m[j] = self.m.addVar(lb=0, ub=NRow, vtype=GRB.INTEGER, name="MT_m_c" + ('[%d]' % j))
        #
        self.DoF_BL = self.m.addVar(lb=1, vtype=GRB.INTEGER, name="DoF_BL")
        self.DoF_RD = self.m.addVar(lb=1, vtype=GRB.INTEGER, name="DoF_RD")
        self.DoM = self.m.addVar(lb=1, vtype=GRB.INTEGER, name="DoM")
        self.Obj = self.m.addVar(lb=1, vtype=GRB.INTEGER, name="Obj")

    def setObjective(self):
        self.m.addConstr(self.DoF_BL - gp.quicksum(self.DoF_init_BL.flatten()) + gp.quicksum(self.CD_x.flatten()) == 0)
        self.m.addConstr(self.DoF_RD - gp.quicksum(self.DoF_init_RD.flatten()) + gp.quicksum(self.CD_y.flatten()) == 0)
        self.m.addConstr(self.DoM - gp.quicksum(self.MT_m.flatten()) == 0)
        self.m.addConstr(self.Obj - self.DoF_BL <= 0)
        self.m.addConstr(self.Obj - self.DoF_RD <= 0)
        self.m.addConstr(self.Obj - self.DoM <= 0)
        self.m.setObjective(self.Obj, GRB.MAXIMIZE)

    def addGenConstrs(self):
        # Define indicator variables
        for ri in range(self.TR):
            for i in range(NRow):
                for j in range(NCol):
                    self.m.addConstr(self.SB_g[ri, i, j] == gp.and_(self.SB_x[ri, i, j], self.SB_y[ri, i, j]))
                    self.m.addConstr(self.SB_w[ri, i, j] + self.SB_x[ri, i, j] + self.SB_y[ri, i, j] - self.SB_g[ri, i, j] == 1)
        for ri in range(self.TR):
            for j in range(NCol):
                self.m.addConstr(self.SB_X[ri, j] == gp.min_(self.SB_x[ri, : ,j].tolist()))
                self.m.addConstr(self.SB_Y[ri, j] == gp.min_(self.SB_y[ri, : ,j].tolist()))
                self.m.addConstr(self.SB_W[ri, j] == gp.max_(self.SB_w[ri, : ,j].tolist()))
                self.m.addConstr(self.MC_X[ri, j] == gp.min_(self.MC_x[ri, : ,j].tolist()))
                self.m.addConstr(self.MC_Y[ri, j] == gp.min_(self.MC_y[ri, : ,j].tolist()))
                self.m.addConstr(self.MC_W[ri, j] == gp.max_(self.MC_w[ri, : ,j].tolist()))

    def MC_RULE(self,
        MC_I_Col_x,
        MC_I_Col_y,
        MC_I_Col_X,
        MC_I_Col_Y,
        MC_I_Col_W,
        MC_O_Col_x,
        MC_O_Col_y,
        CD_Col_x,
        CD_Col_y):
        # Introduce 0-1 indicator variables for each input column
        # Constraints for defining attribute-propagation through MC
        self.m.addConstr(gp.quicksum(MC_O_Col_x) + NRow * MC_I_Col_W <= NRow)
        self.m.addConstr(gp.quicksum(MC_O_Col_x) + gp.quicksum(MC_I_Col_x) - N_Br * MC_I_Col_X <= N_IO - N_Br)
        self.m.addConstr(gp.quicksum(MC_O_Col_x) + gp.quicksum(MC_I_Col_x) - N_IO * MC_I_Col_X >= 0)
        self.m.addConstr(gp.quicksum(MC_O_Col_y) + NRow * MC_I_Col_W <= NRow)
        self.m.addConstr(gp.quicksum(MC_O_Col_y) + gp.quicksum(MC_I_Col_y) - N_Br * MC_I_Col_Y <= N_IO - N_Br)
        self.m.addConstr(gp.quicksum(MC_O_Col_y) + gp.quicksum(MC_I_Col_y) - N_IO * MC_I_Col_Y >= 0)
        # Constraints for canceling impact by consuming DoF
        self.m.addConstr(gp.quicksum(MC_O_Col_x) - NRow * MC_I_Col_X - CD_Col_y == 0)
        self.m.addConstr(gp.quicksum(MC_O_Col_y) - NRow * MC_I_Col_Y - CD_Col_x == 0)

    def Match_RULE(self,
        MC_I_Col_x,
        MC_I_Col_y,
        MC_I_Col_g,
        MC_O_Col_x,
        MC_O_Col_y,
        MC_O_Col_g,
        MT_Col_n,
        MT_Col_m):
        self.m.addConstr(MT_Col_n == 
            gp.quicksum(MC_I_Col_x) + gp.quicksum(MC_I_Col_y) - gp.quicksum(MC_I_Col_g) + 
            gp.quicksum(MC_O_Col_x) + gp.quicksum(MC_O_Col_y) - gp.quicksum(MC_O_Col_g) - NRow)
        self.m.addConstr(MT_Col_m == gp.max_(MT_Col_n, 0))

    def addConstrs_Start_Round(self):
        # Starting round do not allow unknown
        for i in range(NRow):
            for j in range(NCol):
                self.m.addConstr(self.SB_x[self.ini_r, i, j] + self.SB_y[self.ini_r, i, j] >= 1)
                self.m.addConstr(self.DoF_init_BL[i, j] + self.SB_y[self.ini_r, i, j] == 1)
                self.m.addConstr(self.DoF_init_RD[i, j] + self.SB_x[self.ini_r, i, j] == 1)

    def addConstrs_Inter_Round(self, ri):
        if WLastMC == 0 and ri == self.TR - 1:
            for j in range(NCol):
                self.m.addConstr(self.CD_x[ri,j] == 0)
                self.m.addConstr(self.CD_y[ri,j] == 0)
                for i in range(NRow):
                    self.m.addConstr(self.MC_x[ri, i, j] - self.SB_x[(ri + 1) % self.TR, i, j] == 0)
                    self.m.addConstr(self.MC_y[ri, i, j] - self.SB_y[(ri + 1) % self.TR, i, j] == 0)
        elif ri in self.F_r:
            for ci in range(NCol):
                self.MC_RULE(
                    self.MC_x[ri, :, ci],
                    self.MC_y[ri, :, ci],
                    self.MC_X[ri, ci],
                    self.MC_Y[ri, ci],
                    self.MC_W[ri, ci],
                    self.SB_x[(ri + 1) % self.TR, :, ci],
                    self.SB_y[(ri + 1) % self.TR, :, ci],
                    self.CD_x[ri,ci],
                    self.CD_y[ri,ci]
                )
        elif ri in self.B_r:
            for ci in range(NCol):
                self.MC_RULE(
                    self.SB_x[(ri + 1) % self.TR, :, ci],
                    self.SB_y[(ri + 1) % self.TR, :, ci],
                    self.SB_X[(ri + 1) % self.TR, ci],
                    self.SB_Y[(ri + 1) % self.TR, ci],
                    self.SB_W[(ri + 1) % self.TR, ci],
                    self.MC_x[ri, :, ci],
                    self.MC_y[ri, :, ci],
                    self.CD_x[ri,ci],
                    self.CD_y[ri,ci]
                )
        else:
            print('Should not reach here: addConstrs_Inter_Round():: ri = %d' % ri)

    def addConstrs_Match_Round(self):
        if WLastMC == 0 and self.mat_r == self.TR - 1:
            for i in range(NRow):
                for j in range(NCol):
                    self.MT_w[i, j] = self.m.addVar(vtype=GRB.BINARY, name="MT_w" + ('[%d,%d]' % (i, j)))
                    self.m.addConstr(self.MT_w[i, j] == gp.or_(self.MC_w[self.mat_r, i, j], self.SB_w[(self.mat_r + 1)%self.TR, i, j]))
            for j in range(NCol):
                self.m.addConstr(self.MT_m[j] + gp.quicksum(self.MT_w[:, j]) == NRow)
                self.m.addConstr(self.CD_x[self.mat_r, j] == 0)
                self.m.addConstr(self.CD_y[self.mat_r, j] == 0)
        else:
            for ci in range(NCol):
                self.Match_RULE(
                    self.MC_x[self.mat_r, :, ci],
                    self.MC_y[self.mat_r, :, ci],
                    self.MC_g[self.mat_r, :, ci],
                    self.SB_x[(self.mat_r + 1) % self.TR, :, ci],
                    self.SB_y[(self.mat_r + 1) % self.TR, :, ci],
                    self.SB_g[(self.mat_r + 1) % self.TR, :, ci],
                    self.MT_n[ci],
                    self.MT_m[ci]
                )
                self.m.addConstr(self.CD_x[self.mat_r, ci] == 0)
                self.m.addConstr(self.CD_y[self.mat_r, ci] == 0)

    def buildModel(self):
        self.addVars()
        self.addGenConstrs()
        for ri in range(self.TR):
            if ri == self.ini_r:
                self.addConstrs_Start_Round()
            if ri == self.mat_r:
                self.addConstrs_Match_Round()
            else:
                self.addConstrs_Inter_Round(ri)
        self.setObjective()
        self.m.write(self.m.modelName + '.lp' )

    def solveModel(self):
        #self.m.setParam(GRB.Param.Threads, 8)
        #self.m.setParam(GRB.Param.PoolSearchMode, 2)
        #self.m.setParam(GRB.Param.PoolSolutions,  5)
        #self.m.setParam(GRB.Param.SolFiles, self.m.ModelName)
        self.m.optimize()

    def writeSol(self):
        if self.m.SolCount > 0:
            if self.m.getParamInfo(GRB.Param.PoolSearchMode)[2] > 0:
                gv = self.m.getVars()
                names = self.m.getAttr('VarName', gv)
                for i in range(self.m.SolCount):
                    self.m.params.SolutionNumber = i
                    xn = self.m.getAttr('Xn', gv)
                    lines = ["{} {}".format(v1, v2) for v1, v2 in zip(names, xn)]
                    with open('{}_{}.sol'.format(self.m.modelName, i), 'w') as f:
                        f.write("# Solution for model {}\n".format(self.m.modelName))
                        f.write("# Objective value = {}\n".format(self.m.PoolObjVal))
                        f.write("\n".join(lines))
            else:
                self.m.write(self.m.modelName + '.sol')
        else:
            print('infeasible')

    def drawSol(self, outfile=None):
        if outfile == None:
            outfile = self.m.modelName + '.sol'
        solFile = open(outfile, 'r')
        Sol = dict()
        for line in solFile:
            if line[0] != '#':
                temp = line
                temp = temp.split()
                Sol[temp[0]] = round(float(temp[1]))
        SB_x_v = np.ndarray(shape=(self.TR, NRow, NCol), dtype='int')
        SB_y_v = np.ndarray(shape=(self.TR, NRow, NCol), dtype='int')
        MC_x_v = np.ndarray(shape=(self.TR, NRow, NCol), dtype='int')
        MC_y_v = np.ndarray(shape=(self.TR, NRow, NCol), dtype='int')
        DoF_init_BL_v = np.ndarray(shape=(NRow, NCol), dtype='int')
        DoF_init_RD_v = np.ndarray(shape=(NRow, NCol), dtype='int')
        CD_x_v = np.ndarray(shape=(self.TR, NCol), dtype='int')
        CD_y_v = np.ndarray(shape=(self.TR, NCol), dtype='int')
        for ri in range(self.TR):
            for i in range(NRow):
                for j in range(NCol):
                    SB_x_v[ri, i, j] = Sol["SB_x" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri)]
                    SB_y_v[ri, i, j] = Sol["SB_y" + ('[%d,%d]' % (i, j)) + "_r" + ('[%d]' % ri)]
        for ri in range(self.TR):
            for i in range(NRow):
                for j in range(NCol):
                    MC_x_v[ri, i, j] = SB_x_v[ri, i, (j + i)%NCol]
                    MC_y_v[ri, i, j] = SB_y_v[ri, i, (j + i)%NCol]
        for i in range(NRow):
            for j in range(NCol):
                DoF_init_BL_v[i, j] = Sol["DoF_init_BL" + ('[%d,%d]' % (i,j))]
                DoF_init_RD_v[i, j] = Sol["DoF_init_RD" + ('[%d,%d]' % (i,j))]
        for ri in range(self.TR):
            for j in range(NCol):
                CD_x_v[ri, j] = Sol["CD_x_c" + ('[%d]' % j) + "_r" + ('[%d]' % ri)]
                CD_y_v[ri, j] = Sol["CD_y_c" + ('[%d]' % j) + "_r" + ('[%d]' % ri)]
        DoF_BL_v = Sol["DoF_BL"]
        DoF_RD_v = Sol["DoF_RD"]
        DoM_v = Sol["DoM"]
        #
        CM = np.ndarray(shape=(2, 2),dtype='object')
        CM[0, 0] = '\\fill[\\UW]'
        CM[0, 1] = '\\fill[\\BW]'
        CM[1, 0] = '\\fill[\\FW]'
        CM[1, 1] = '\\fill[\\CW]'
        if NRow == 4:
            HO = NRow
            WO = NCol
        else:
            HO = NRow
            WO = NCol // 2
        ini_d1 = 0
        ini_d2 = 0
        for i in range(NRow):
            for j in range(NCol):
                ini_d1 += DoF_init_BL_v[i,j]
                ini_d2 += DoF_init_RD_v[i,j]
        fid = open(outfile + '.tex', 'w')
        fid.write(
            '\\documentclass{standalone}' + '\n'
            '\\usepackage[usenames,dvipsnames]{xcolor}' + '\n'
            '\\usepackage{amsmath,amssymb,mathtools}' + '\n'
            '\\usepackage{tikz,calc,pgffor}' + '\n'
            '\\usepackage{xspace}' + '\n'
            '\\usetikzlibrary{crypto.symbols,patterns,calc}' + '\n'
            '\\tikzset{shadows=no}' + '\n'
            '\\input{macro}' + '\n')
        fid.write('\n\n')
        fid.write(
            '\\begin{document}' + '\n' +
            '\\begin{tikzpicture}[scale=0.2, every node/.style={font=\\boldmath\\bf}]' + '\n'
            '\\everymath{\\scriptstyle}' + '\n'
            '\\tikzset{edge/.style=->, >=stealth, arrow head=8pt, thick};' + '\n')
        fid.write('\n\n')
        #
        for r in range(self.TR):
            CD_BL = 0
            CD_RD = 0
            for i in range(NCol):
                CD_BL += CD_x_v[r, i]
                CD_RD += CD_y_v[r, i]
            O = 0
            ## SB
            fid.write('\\begin{scope}[yshift =' + str(- r * (NRow + HO))+' cm, xshift =' + str(O * (NCol + WO))+' cm]'+'\n')
            for i in range(NRow):
                row = NRow - 1 - i
                for j in range(NCol):
                    col = j
                    fid.write(CM[SB_x_v[r,i,j], SB_y_v[r,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
            fid.write('\\draw (0,0) rectangle (' + str(NCol) + ',' + str(NRow) + ');' + '\n')
            for i in range(1, NRow):
                fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCol) + ',' + str(0) + ');' + '\n')
            for i in range(1, NCol):
                fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NRow) + ');' + '\n')
            fid.write('\\path (' + str(NCol//2) + ',' + str(NRow + 0.5) + ') node {\\scriptsize$\\SB^' + str(r) + '$};'+'\n')
            if r in self.B_r:
                fid.write('\\draw[edge, <-] (' + str(NCol) + ',' + str(NRow//2) + ') -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(' + str(WO) + ',' + '0);' + '\n')
            else:
                fid.write('\\draw[edge, ->] (' + str(NCol) + ',' + str(NRow//2) + ') -- node[above] {\\tiny SB} node[below] {\\tiny SR} +(' + str(WO) + ',' + '0);' + '\n')
            if r == self.ini_r:
                fid.write('\\path (' + str(NCol//2) + ',' + str(-0.8) + ') node {\\scriptsize$(+' + str(ini_d1) + '~\\DoFF,~+' + str(ini_d2) + '~\\DoFB)$};'+'\n')
                fid.write('\\path (' + str(-2) + ',' + str(0.8) + ') node {\\scriptsize$\\StENC$};'+'\n')
            fid.write('\n'+'\\end{scope}'+'\n')
            fid.write('\n\n')

            O = O + 1
            ## MC
            fid.write('\\begin{scope}[yshift =' + str(- r * (NRow + HO))+' cm, xshift =' +str(O * (NCol + WO))+' cm]'+'\n')
            for i in range(NRow):
                row = NRow - 1 - i
                for j in range(NCol):
                    col = j
                    fid.write(CM[MC_x_v[r,i,j], MC_y_v[r,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
            fid.write('\\draw (0,0) rectangle (' + str(NCol) + ',' + str(NRow) + ');' + '\n')
            for i in range(1, NRow):
                fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCol) + ',' + str(0) + ');' + '\n')
            for i in range(1, NCol):
                fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NRow) + ');' + '\n')
            fid.write('\\path (' + str(NCol//2) + ',' + str(NRow + 0.5) + ') node {\\scriptsize$\\MC^' + str(r) + '$};'+'\n')
            op = 'MC'
            if r == self.TR - 1 and WLastMC == 0:
                op = 'I'
            if r in self.B_r:
                fid.write('\\draw[edge, <-] (' + str(NCol) + ',' + str(NRow//2) + ') -- node[above] {\\tiny ' + op + '} +(' + str(WO) + ',' + '0);' + '\n')
            if r in self.F_r:
                fid.write('\\draw[edge, ->] (' + str(NCol) + ',' + str(NRow//2) + ') -- node[above] {\\tiny ' + op + '} +(' + str(WO) + ',' + '0);' + '\n')
            if r == self.mat_r:
                fid.write('\\draw[edge, -] (' + str(NCol) + ',' + str(NRow//2) + ') -- node[above] {\\tiny ' + op + '} +(' + str(WO) + ',' + '0);' + '\n')
                fid.write('\\draw[edge, ->] (' + str(NCol) + ',' + str(NRow//2) + ') --  +(' + str(WO//2) + ',' + '0);' + '\n')
                fid.write('\\draw[edge, ->] (' + str(NCol + WO) + ',' + str(NRow//2) + ') --  +(' + str(-WO//2) + ',' + '0);' + '\n')

                fid.write('\\path (' + str(NCol + WO//2) + ',' + str(-0.8) + ') node {\\scriptsize Match};' + '\n')
                fid.write('\\path (' + str(-2) + ',' + str(0.1) + ') node {\\scriptsize$\\EndFwd$};' + '\n')
                fid.write('\\path (' + str(NCol + WO + NCol + 2) + ',' + str(0.1) + ') node {\\scriptsize$\\EndBwd$};' + '\n')
            else:
                fid.write('\\path (' + str((NCol + WO) - WO//2) + ',' + str(-0.8) + ') node {\\scriptsize$ (-' + str(CD_BL) + '~\\DoFF,~-' + str(CD_RD) + '~\\DoFB)$};'+'\n')
            fid.write('\n'+'\\end{scope}'+'\n')
            fid.write('\n\n')

            O = O + 1
            ## SB r+1
            fid.write('\\begin{scope}[yshift =' + str(- r * (NRow + HO))+' cm, xshift =' +str(O * (NCol + WO))+' cm]'+'\n')
            for i in range(NRow):
                row = NRow - 1 - i
                for j in range(NCol):
                    col = j
                    fid.write(CM[SB_x_v[(r+1)%self.TR,i,j], SB_y_v[(r+1)%self.TR,i,j]] + ' ('+str(col)+','+str(row)+') rectangle +(1,1);'+'\n')
            fid.write('\\draw (0,0) rectangle (' + str(NCol) + ',' + str(NRow) + ');' + '\n')
            for i in range(1, NRow):
                fid.write('\\draw (' + str(0) + ',' + str(i) + ') rectangle (' + str(NCol) + ',' + str(0) + ');' + '\n')
            for i in range(1, NCol):
                fid.write('\\draw (' + str(i) + ',' + str(0) + ') rectangle (' + str(0) + ',' + str(NRow) + ');' + '\n')
            fid.write('\\path (' + str(NCol//2) + ',' + str(NRow + 0.5) + ') node {\\scriptsize$\\SB^' + str((r+1)%self.TR) + '$};'+'\n')
            fid.write('\n'+'\\end{scope}'+'\n')
            fid.write('\n\n')
        ## Final
        fid.write('\\begin{scope}[yshift =' + str(- self.TR * (NRow + HO) + HO)+' cm, xshift =' +str(2 * (NCol + WO))+' cm]'+'\n')
        fid.write(
            '\\node[draw, thick, rectangle, text width=6.5cm, label={[shift={(-2.8,-0)}]\\footnotesize Config}] at (-7, 0) {' + '\n'
    	    '{\\footnotesize' + '\n'
    	    '$\\bullet~(\\varInitBL,~\\varInitRD)~=~(+' + str(ini_d1) + '~\\DoFF,~+' + str(ini_d2) + '~\\DoFB)~$' + '\\\ \n'
    	    '$\\bullet~(\\varDoFBL,~\\varDoFRD,~\\varDoM)~=~(+' + 
            str(int(DoF_BL_v)) + '~\\DoFF,~+' + 
            str(int(DoF_RD_v)) + '~\\DoFB,~+' + 
            str(int(DoM_v )) + '~\\DoM)$' + '\n'
    	    '}' + '\n'
    	    '};' + '\n'
            )
        fid.write('\n'+'\\end{scope}'+'\n')
        fid.write('\n\n')
        fid.write('\\end{tikzpicture}'+'\n\n'+'\\end{document}')
        fid.close()
        system("pdflatex -output-directory=" + self.fpath + ' ' + outfile + ".tex") 

def search(fpath='./', TR=3):
    resultf = open(fpath + '/Result_r' + str(TR) + '.txt', 'w+')
    resultf.write('%6s, %6s, %6s: %6s (%6s, %6s, %6s); %10s secs\n' %
                 ('TR', 'ini_r', 'mat_r', 'Obj', 'DoF1', 'DoF2', 'DoM', 'Time'))
    best = 0
    start = time.time()
    for ini_r in list(range(TR//2, TR)) + list(range(TR//2 - 1, -1, -1)):
        for mat_r in range(TR):
            if mat_r != ini_r:
                startt = time.time()
                searcher = MitM_Attack_Searcher(TR, ini_r, mat_r, fpath)
                searcher.buildModel()
                searcher.solveModel()
                elapsedt = (time.time() - startt)
                if searcher.m.SolCount > 0:
                    searcher.writeSol()
                    #searcher.drawSol()
                    Obj  = int(round(searcher.Obj.X))
                    DoF1 = int(round(searcher.DoF_BL.X))
                    DoF2 = int(round(searcher.DoF_RD.X))
                    DoM  = int(round(searcher.DoM.X))
                    if Obj > best:
                        best = Obj
                    resultf.write('%6d, %6d, %6d: %6d (%6d, %6d, %6d); %10f secs\n' % 
                                    (TR, ini_r, mat_r, Obj, DoF1, DoF2, DoM, elapsedt))
                    resultf.flush()
                else:
                    resultf.write('%6d, %6d, %6d: %6s (%6s, %6s, %6s); %10f secs\n' % 
                                    (TR, ini_r, mat_r, '-', '-', '-', '-', elapsedt))
                    resultf.flush()
    elapsed = (time.time() - start)
    resultf.write('Best Obj: ' + str(best) + '\n')
    resultf.write('Total time (secs): ' + str(elapsed) + '\n')
    resultf.close()


if __name__ == "__main__":
    search(sys.argv[1], int(sys.argv[2]))

