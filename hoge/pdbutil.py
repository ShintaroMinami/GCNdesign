"""
The MIT License (MIT)

Copyright (c) 2020 Shintaro Minami.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import sys
import numpy as np

class ProteinBackbone:
    """
    Simple class for handling protein backbone structure.

    Attributes
    ----------
    naa : int
        Number of residues.
    coord : numpy float matrix (naa, 6, 3)
        3D Coordinates of 6 backbone atoms (N,CA,C,O,CB,H).
    exists : numpy bool matrix (naa, 6)
        Existence of the coodinates.
    resname : numpy str vector (naa)
        Residue name.
    iaa2org : numpy str vector (naa)
        Original chain ID and residue number.
    dihedral : numpy float matrix (naa, 3)
        Dihedral angles (phi, psi, omega).
    distmat : numpy float matrix (naa, naa)
        Distance matrix.
    """

    def __init__(self, length=0, file=None, copyfrom=None):
        """
        Parameters
        ----------
        file : str
            Path to the PDB file.
        copyfrom : instance of this class (ProteinBackbone).
            Original instance to be copied.
        length : int
            Number of residues.
        """
        self.atom2id = {'N':0, 'CA':1, 'C':2, 'O':3, 'CB':4, 'H':5}
        self.id2atom = ['N', 'CA', 'C', 'O', 'CB', 'H']
        self.param = {'angle_N_CA_CB':np.deg2rad(110.6), 'angle_CB_CA_C':np.deg2rad(110.6),
                      'angle_C_N_H':np.deg2rad(123.0), 'angle_N_C_O':np.deg2rad(122.7),
                       'dhdrl_C_N_CA_CB':np.deg2rad(-124.4), 'dhdrl_N_C_CA_CB':np.deg2rad(121.5),
                       'dhdrl_CA_C_N_H':np.deg2rad(0.0), 'dhdrl_CA_N_C_O':np.deg2rad(0.0),
                       'length_CC':1.54, 'length_CO':1.24, 'length_NH':1.00}
        if file is not None:
            self.file = file
            self.readpdb(self.file)
            self.addO()
        elif copyfrom is not None:
            self.naa = copyfrom.naa
            self.coord = copyfrom.coord
            self.exists = copyfrom.exists
            self.resname = copyfrom.resname
            self.iaa2org = copyfrom.iaa2org
        elif length > 0:
            self.naa = length
            self.coord = np.zeros((self.naa, len(self.atom2id), 3), dtype=np.float)
            self.exists = np.ones((self.naa, len(self.atom2id)), dtype=np.bool)
            self.exists[:,self.atom2id['CB']] = False
            self.exists[:,self.atom2id['H']] = False
            self.resname = ['NON']*self.naa
            self.iaa2org = ['A0000']*self.naa

    def __getitem__(self, ids):
        return self.coord[ids]

    def __setitem__(self, ids, val):
        self.coord[ids] = val

    def __len__(self):
        return self.naa

    ## calc dihedral angle ##
    def calc_dihedral(self):
        self.dihedral = np.zeros((self.naa, 3), dtype=np.float)
        for iaa in range(self.naa):
            if (iaa > 0) and (self.exists[iaa-1][self.atom2id['C']] == True):
                self.dihedral[iaa][0] = xyz2dihedral(self.coord[iaa-1][self.atom2id['C']],
                                                     self.coord[iaa][self.atom2id['N']],
                                                     self.coord[iaa][self.atom2id['CA']],
                                                     self.coord[iaa][self.atom2id['C']])
            if (iaa < self.naa-1) and (self.exists[iaa+1][self.atom2id['N']] == True):
                self.dihedral[iaa][1] = xyz2dihedral(self.coord[iaa][self.atom2id['N']],
                                                     self.coord[iaa][self.atom2id['CA']],
                                                     self.coord[iaa][self.atom2id['C']],
                                                     self.coord[iaa+1][self.atom2id['N']])
            if (iaa < self.naa-1) and (self.exists[iaa+1][self.atom2id['CA']] == True):
                self.dihedral[iaa][2] = xyz2dihedral(self.coord[iaa][self.atom2id['CA']],
                                                     self.coord[iaa][self.atom2id['C']],
                                                     self.coord[iaa+1][self.atom2id['N']],
                                                     self.coord[iaa+1][self.atom2id['CA']])

    ## delete residues ##
    def delete(self, position, length):
        naa_org = self.naa
        coord_org = self.coord
        exists_org = self.exists
        resname_org = self.resname
        iaa2org_org = self.iaa2org
        self.naa = self.naa - length
        self.coord = np.zeros((self.naa, len(self.atom2id), 3), dtype=np.float)
        self.exists = np.zeros((self.naa, len(self.atom2id)), dtype=np.bool)
        self.resname = ['NAN']*self.naa
        self.iaa2org = ['A0000']*self.naa
        iaa_new = 0
        for iaa in range(naa_org):
            if position <= iaa < position+length: continue
            self.coord[iaa_new] = coord_org[iaa]
            self.exists[iaa_new] = exists_org[iaa]
            self.resname[iaa_new] = resname_org[iaa]
            self.iaa2org[iaa_new] = iaa2org_org[iaa]
            iaa_new += 1

    ## insert blank residues ##
    def insert_blank(self, position, length, chain='A', resname='INS'):
        naa_org = self.naa
        coord_org = self.coord
        exists_org = self.exists
        resname_org = self.resname
        iaa2org_org = self.iaa2org
        self.naa = self.naa + length
        self.coord = np.zeros((self.naa, len(self.atom2id), 3), dtype=np.float)
        self.exists = np.ones((self.naa, len(self.atom2id)), dtype=np.bool)
        self.exists[:,self.atom2id['CB']] = False
        self.exists[:,self.atom2id['H']] = False
        self.resname = [resname]*self.naa
        self.iaa2org = [chain+'0000']*self.naa
        iaa_new = 0
        for iaa in range(naa_org):
            if iaa == position:
                for i in range(length):
                    iaa_new += 1
            self.coord[iaa_new] = coord_org[iaa]
            self.exists[iaa_new] = exists_org[iaa]
            self.resname[iaa_new] = resname_org[iaa]
            self.iaa2org[iaa_new] = iaa2org_org[iaa]
            iaa_new += 1

    ## insert fragment ##
    def insert(self, position, insertion):
        length = len(insertion)
        self.insert_blank(position, length)
        self.coord[position:position+length] = insertion.coord
        self.exists[position:position+length] = insertion.exists
        self.resname[position:position+length] = insertion.resname
        self.iaa2org[position:position+length] = insertion.iaa2org

    ## add vitual H atoms ##
    def addH(self, force=False):
        for iaa in range(1,len(self.coord)):
            if ((self.exists[iaa][self.atom2id['H']] == True) and (force==False)): continue
            nh = zmat2xyz(self.param['length_NH'],
                          self.param['angle_C_N_H'],
                          self.param['dhdrl_CA_C_N_H'],
                          self.coord[iaa-1][self.atom2id['CA']],
                          self.coord[iaa-1][self.atom2id['C']],
                          self.coord[iaa][self.atom2id['N']])
            self.coord[iaa][self.atom2id['H']][0] = nh[0]
            self.coord[iaa][self.atom2id['H']][1] = nh[1]
            self.coord[iaa][self.atom2id['H']][2] = nh[2]
            self.exists[iaa][self.atom2id['H']] = True

    ## add virtual O atoms ##
    def addO(self, force=False):
        for iaa in range(len(self.coord)-1):
            if ((self.exists[iaa][self.atom2id['O']] == True) and (force==False)): continue
            co = zmat2xyz(self.param['length_CO'],
                          self.param['angle_N_C_O'],
                          self.param['dhdrl_CA_N_C_O'],
                          self.coord[iaa+1][self.atom2id['CA']],
                          self.coord[iaa+1][self.atom2id['N']],
                          self.coord[iaa][self.atom2id['C']])
            self.coord[iaa][self.atom2id['O']][0] = co[0]
            self.coord[iaa][self.atom2id['O']][1] = co[1]
            self.coord[iaa][self.atom2id['O']][2] = co[2]
            self.exists[iaa][self.atom2id['O']] = True

    ## add virtual CB atoms ##
    def addCB(self, force=False):
        for iaa in range(len(self.coord)):
            if ((self.exists[iaa][self.atom2id['CB']] == True) and (force==False)): continue
            cb1 = zmat2xyz(self.param['length_CC'],
                           self.param['angle_N_CA_CB'],
                           self.param['dhdrl_C_N_CA_CB'],
                           self.coord[iaa][self.atom2id['C']],
                           self.coord[iaa][self.atom2id['N']],
                           self.coord[iaa][self.atom2id['CA']])
            cb2 = zmat2xyz(self.param['length_CC'],
                           self.param['angle_CB_CA_C'],
                           self.param['dhdrl_N_C_CA_CB'],
                           self.coord[iaa][self.atom2id['N']],
                           self.coord[iaa][self.atom2id['C']],
                           self.coord[iaa][self.atom2id['CA']])
            cb = (cb1 + cb2)/2.0
            self.coord[iaa][self.atom2id['CB']][0] = cb[0]
            self.coord[iaa][self.atom2id['CB']][1] = cb[1]
            self.coord[iaa][self.atom2id['CB']][2] = cb[2]
            self.exists[iaa][self.atom2id['CB']] = True

    ## distance matrix ##
    def calc_distmat(self, atomtype='CA'):
        points = self.coord[:,self.atom2id[atomtype],:]
        self.distmat = np.sqrt( np.sum((points[np.newaxis,:,:] - points[:,np.newaxis,:])**2, axis=2) )

    ## get nearest N residues ##
    def get_nearestN(self, N, atomtype='CA', distmat=True, rm_self=True):
        if distmat:
            self.calc_distmat(atomtype=atomtype)
        if rm_self:
            N = N+1
        args_topN_unsorted = np.argpartition(self.distmat, N)[:,:N]
        args_topN_sorted = np.ndarray((self.distmat.shape[0], N), dtype=np.int)
        for i in range(self.distmat.shape[0]):
            vals = self.distmat[i][args_topN_unsorted[i]]
            indices = np.argsort(vals)
            args_topN_sorted[i] = args_topN_unsorted[i][indices]
        if rm_self:
            args_topN_sorted = args_topN_sorted[:,1:]
        return args_topN_sorted

    ## print pdb format ##
    def printpdb(self, file=sys.stdout, chain=None, start=None, region=None):
        icount = 0
        if region is not None:
            outrange = range(region[0], region[1]+1)
        else:
            outrange = range(len(self.coord))
        for iaa in outrange:
            if chain is None:
                chain = self.iaa2org[iaa][0:1]
            if start is None:
                resnum = int(self.iaa2org[iaa][1:5])
            else:
                resnum = int(start) + iaa - outrange[0]
            for iatom in range(len(self.id2atom)):
                if(self.exists[iaa][iatom] == False): continue
                icount += 1
                file.write("ATOM%7d  %-3s %3s %s%4d    %8.3f%8.3f%8.3f\n"
                           % (icount, self.id2atom[iatom], self.resname[iaa],
                              chain, resnum,
                              self.coord[iaa][iatom][0],
                              self.coord[iaa][iatom][1],
                              self.coord[iaa][iatom][2]))

    ## read pdb file ##
    def readpdb(self, file):
        lines = open(file, "r").read().splitlines()
        # exists protein length
        self.naa = 0
        self.org2iaa = {}
        for l in lines:
            (header, atomtype, resname, chain, iaa_org) = (l[0:4], l[12:16].strip(), l[17:20], l[21:22], l[22:27])
            if not ((header == "ATOM") and (atomtype == 'CA')) : continue
            self.org2iaa[(chain+iaa_org)] = self.naa
            self.naa += 1
        # read ATOM lines
        self.coord = np.zeros((self.naa, len(self.atom2id), 3), dtype=np.float)
        self.exists = np.zeros((self.naa, len(self.atom2id)), dtype=np.bool)
        self.resname = ['NAN']*self.naa
        self.iaa2org = ['A0000 ']*self.naa
        for l in lines:
            (header, atomtype, resname, chain, iaa_org) = (l[0:4], l[12:16].strip(), l[17:20], l[21:22], l[22:27])
            if not (header == "ATOM"): continue
            if atomtype not in self.atom2id: continue
            org = (chain+iaa_org)
            iaa = self.org2iaa.get(org)
            if iaa is None: continue
            id_atom = self.atom2id[atomtype]
            coord = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
            self.coord[iaa][id_atom][0] = coord[0]
            self.coord[iaa][id_atom][1] = coord[1]
            self.coord[iaa][id_atom][2] = coord[2]
            self.exists[iaa][id_atom] = True
            self.resname[iaa] = resname
            self.iaa2org[iaa] = org
        return



#### Functions ####
def zmat2xyz(bond, angle, dihedral, one, two , three):
    oldvec = np.ones(4, dtype=np.float)
    oldvec[0] = bond * np.sin(angle) * np.sin(dihedral)
    oldvec[1] = bond * np.sin(angle) * np.cos(dihedral)
    oldvec[2] = bond * np.cos(angle)
    mat = viewat(three, two, one)
    newvec = np.zeros(4, dtype=np.float)
    for i in range(4):
        for j in range(4):
            newvec[i] += mat[i][j] * oldvec[j]
    # return
    return newvec

def viewat(p1, p2, p3):
    # vector #
    p12 = p2 - p1
    p13 = p3 - p1
    # normalize #
    z = p12 / np.linalg.norm(p12)
    # crossproduct #
    x = np.cross(p13, p12)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    # transpation matrix
    mat = np.zeros((4, 4), dtype=np.float)
    for i in range(3):
        mat[i][0] = x[i]
        mat[i][1] = y[i]
        mat[i][2] = z[i]
        mat[i][3] = p1[i]
    mat[3][3] = 1.0
    # return
    return mat

def xyz2dihedral(p1, p2, p3, p4):
    # small val #
    eps = 0.0000001
    # bond vector
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    # perpendicular vector #
    perp123 = np.cross(v1, v2)
    perp234 = np.cross(v2, v3)
    perp123 /= np.linalg.norm(perp123)
    perp234 /= np.linalg.norm(perp234)
    # scalar product #
    scp = np.dot(perp123, perp234)
    scp = scp - eps if (1-eps < scp < 1+eps) else scp
    scp = scp + eps if (-1-eps < scp < -1+eps) else scp
    # absolute angle #
    angle = np.rad2deg( np.arccos(scp) )
    # return #
    return angle if np.dot(v1, perp234) > 0 else -angle
