
def expand_nums(numlist):
    newlist = []
    for l in numlist:
        if '-' in l:
            ini, end = [int(i) for i in l.split('-')]
            newlist += list(range(ini, end+1))
        else:
            newlist.append(int(l))
    return set(newlist)


def fix_native_resfile(lines_resfile, resnums=[]):
    lines_fixed = ""
    for l in lines_resfile.split('\n'):
        l = l.strip()
        if 'PIKAA' in l:
            i, c, _, aas, _, org = l.split()
            if int(i) in resnums:
                aas = org
            lines_fixed += "{:5d} {} PIKAA  {:20s} # {}\n".format(int(i), c, aas, org)
        elif 'start' in l:
            lines_fixed += "{}\n".format(l)
    return lines_fixed
