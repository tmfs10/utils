
# Taken from https://www.neb.com/tools-and-resources/selection-charts/type-iis-restriction-enzymes
# IIS restriction enzyme -> [recognition sequence, overhang length]
iis_enzymes = {
        'acul' : ['ctgaag', 2],
        'alwl' : ['ggatc', 1],
        'bbsl' : ['gaagac', 4],
        'bbsl-hf': ['gaagac', 4],
        'bbvl': ['gcagc', 4],
        'bccl': ['ccatc', 1],
        'bceal': ['acggc', 2],
        'bcivl': ['gtatcc', 1],
        'bsai': ['ggtctc', 4],
        'bsmbl': ['cgtctc', 4],
        'esp3i': ['cgtctc', 4],
        'sapi': ['gctcttc', 3],
        }


#Ala    A    Alanine
#Arg    R    Arginine
#Asn    N    Asparagine
#Asp    D    Aspartic acid
#Cys    C    Cysteine
#Gln    Q    Glutamine
#Glu    E    Glutamic acid
#Gly    G    Glycine
#His    H    Histidine
#Ile    I    Isoleucine
#Leu    L    Leucine
#Lys    K    Lysine
#Met    M    Methionine
#Phe    F    Phenylalanine
#Pro    P    Proline
#Pyl    O    Pyrrolysine -- not including here
#Ser    S    Serine
#Sec    U    Selenocysteine -- not including here
#Thr    T    Threonine
#Trp    W    Tryptophan
#Tyr    Y    Tyrosine
#Val    V    Valine
#Asx    B    Aspartic acid or Asparagine
#Glx    Z    Glutamic acid or Glutamine
#Xaa    X    Any amino acid
#Xle    J    Leucine or Isoleucine
#TERM   *    termination codon
class AminoAcid:
    letters = ['a', 'r', 'n', 'd', 'c', 'q', 'e', 'g', 'h', 'i', 'l', 'k', 'm', 'f', 'p', 's', 't', 'w', 'y', 'v', 'x', 'b', 'z', 'j', '*']
    names = ['ala', 'arg', 'asn' , 'asp', 'cys', 'gln', 'glu', 'gly', 'his', 'ile', 'leu', 'lys', 'met', 'phe', 'pro', 'ser', 'thr', 'trp', 'tyr', 'val']
    @staticmethod
    def aa_letters():
        return AminoAcid.letters[:20]

    @staticmethod
    def aa_letter_codes():
        return AminoAcid.letters

    prop_groups = {
            'short_chains' : ['a', 'g'],
            'acidic' : ['d', 'e', 'b', 'z'],
            'basic' : ['k', 'r', 'h'],
            'amine' : ['n', 'q', 'b'],
            'sulfide' : ['c', 'm'],
            'alcohol' : ['s', 't', 'y'],
            'alphatic' : ['i', 'l', 'v', 'm', 'a', 'j'],
            'aromatic' : ['f', 'y', 'w', 'h'],
            'proline' : ['p'],
            'polar' : ['d', 'e', 'n', 'q', 'r', 'k', 'h', 'y', 'c', 's', 't', 'b', 'z'],
            'hydrophobic' : ['g', 'a', 'f', 'w', 'p', 'i', 'l', 'v', 'm', 'j'],
            }

    to_dna = {
            'f' : ['ttt', 'ttc', 'tta', 'ttg'], # Phenylalanine (Phe)
            'l' : ['ctt', 'ctc', 'cta', 'ctg'], # Leucine (Leu)
            'i' : ['att', 'atc', 'ata'], # Isoleucine (Ile)
            'm' : ['atg'], # Methionine (Met)
            'v' : ['gtt', 'gtc', 'gta', 'gtg'], # Valine (Val)
            's' : ['tct', 'tcc', 'tca', 'tcg', 'agt', 'agc'], # Serine (Ser)
            'p' : ['cct', 'ccc', 'cca', 'ccg'], # Proline (Pro)
            't' : ['act', 'acc', 'aca', 'acg'], # Threonine (Thr)
            'a' : ['gct', 'gcc', 'gca', 'gcg'], # Alanine (Ala)
            'y' : ['tat', 'tac'], # Tyrosine (Tyr)
            'h' : ['cat', 'cac'], # Histidine (His)
            'q' : ['caa', 'cag'], # Glutamine (Gln)
            'n' : ['aat', 'aac'], # Asparagine (Asn)
            'k' : ['aaa', 'aag'], # Lysine (Lys)
            'd' : ['gat', 'gac'], # Aspartic acid
            'e' : ['gaa', 'gag'], # Glutamic acid
            'c' : ['tgt', 'tgc'], # Cysteine (Cys)
            'w' : ['tgg'], # Tryptophan (Trp)
            'r' : ['cgt', 'cgc', 'cga', 'cgg', 'aga', 'agg'], # Arginine (Arg)
            'g' : ['ggt', 'ggc', 'gga', 'ggg'], # Glycine (Gly)
            'b' : ['gat', 'gac', 'aat', 'aac'],
            'z' : ['gaa', 'gag', 'caa', 'cag'],
            'j' : ['att', 'atc', 'ata', 'aaa', 'aag'],
            '*' : ['taa', 'tag', 'tga'], # stop codon
            }

    prob_matrix = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
        [0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
        [0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
        [1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21, 1./21,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,],
        ]

    blosum62 = {
            'a': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0, -4],
            'r': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1, -4],
            'n': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1, -4],
            'd': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1, -4],
            'c': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],
            'q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1, -4],
            'e': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4],
            'g': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1, -4],
            'h': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1, -4],
            'i': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1, -4],
            'l': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1, -4],
            'k': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1, -4],
            'm': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1, -4],
            'f': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1, -4],
            'p': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2, -4],
            's': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0, -4],
            't': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0, -4],
            'w': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2, -4],
            'y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1, -4],
            'v': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1, -4],
            'x': [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1, -4],
            'b': [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1, -4],
            'z': [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1, -4],
            '*': [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1],
            }


#A   Adenine
#C   Cytosine
#G   Guanine
#T (or U)    Thymine (or Uracil)
#R   A or G
#Y   C or T
#S   G or C
#W   A or T
#K   G or T
#M   A or C
#B   C or G or T
#D   A or G or T
#H   A or C or T
#V   A or C or G
#N   any base
dna_bases = ['a', 'c', 'g', 't', 'r', 'y', 's', 'w', 'k', 'm', 'b', 'd', 'h', 'v', 'n']

dna_to_pr_matrix = [
        [1, 0, 0, 0,],
        [0, 1, 0, 0,],
        [0, 0, 1, 0,],
        [0, 0, 0, 1,],
        [0.5, 0, 0.5, 0,],
        [0, 0.5, 0, 0.5,],
        [0, 0.5, 0.5, 0,],
        [0.5, 0, 0, 0.5,],
        [0, 0, 0.5, 0.5,],
        [0.5, 0.5, 0, 0,],
        [0, 0.33, 0.34, 0.33,],
        [0.33, 0, 0.33, 0.34,],
        [0.33, 0.34, 0, 0.33,],
        [0.34, 0.33, 0.33, 0,],
        [0.25, 0.25, 0.25, 0.25,],
        ]

dna_to_index = {dna_bases[i] : i for i in range(len(dna_bases))}
dna_to_aa = {dna : aa for aa in AminoAcid.to_dna for dna in AminoAcid.to_dna[aa]}

# numbered from 1
kabat_lcdr = [[24, 34], [50, 56], [89, 97]]
kabat_hcdr = [[31, 35], [50, 65], [95, 102]]

