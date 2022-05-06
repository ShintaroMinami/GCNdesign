import re

def add_chain_id(l, default_aa='A'):
    if re.fullmatch(r'[0-9]+', l):
        l = l + default_aa
    if re.fullmatch(r'[0-9]+-[0-9]+', l):
        l1, l2 = l.split('-')
        l = '-'.join([l1+default_aa, l2+default_aa])
    if re.fullmatch(r'@', l):
        l = l + default_aa
    return l

def expand_nums(numlist, min_aa_num=1, max_aa_num=2000):
    numlist = [add_chain_id(l) for l in numlist] # if no chain ID, considered as 'A'
    newlist = []
    for l in numlist:
        if '@' in l:
            newlist += [str(i)+l[-1] for i in list(range(min_aa_num,max_aa_num+1))]
        elif '-' in l:
            ini, end = [re.match(r'[0-9]+', s).group() for s in l.split('-')]
            newlist += [str(i)+l[-1] for i in list(range(int(ini),int(end)+1))]
        else:
            newlist.append(l)
    return set(newlist)


def fix_native_resfile(lines_resfile, resnums=[], keeptype='NATRO'):
    lines_fixed = ""
    for l in lines_resfile.split('\n'):
        l = l.strip()
        if 'PIKAA' in l:
            i, c, _, aas, _, org = l.split()
            if str(i)+c in resnums:
                aas = org
                lines_fixed += "{:5d} {} {}  {:20s} # {}\n".format(int(i), c, keeptype, '', org)
            else:
                lines_fixed += "{:5d} {} {}  {:20s} # {}\n".format(int(i), c, 'PIKAA', aas, org)
        elif 'start' in l:
            lines_fixed += "{}\n".format(l)
    return lines_fixed
