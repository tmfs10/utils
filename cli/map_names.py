#!/usr/bin/env python

import sys
import argparse

"""
In normal mode, maps names. Prints the line in which the domain name is. If it can't find the domain name, it skips the line. If the domain maps to multiple names, then it replicates the line for each range name.
"""

"""
parser = argparse.ArgumentParser()
parser.add_argument(
    "gene_name_mapping_file",
    metavar="gene_name_mapping_file",
    type=str,
    nargs=1,
    help="mapping file for gene"
	"""


if len(sys.argv) < 4:
	print >> sys.stderr, "Usage", sys.argv[0], "<gene name mapping file> <forward?> <col> [<print mismatch?>] < target file"
	sys.exit(1)

gene_name_mapping_file = sys.argv[1]
forward = int(sys.argv[2])
col = int(sys.argv[3])-1
print_mismatch = int(sys.argv[4]) != 0 if len(sys.argv) >= 5 else True

m = {}

if len(sys.argv) >= 6:
	print_domain = (int(sys.argv[6]) == 1)

def map_name(m, line, s):
	if len(line) == 0:
		print s
		return
	name = line[0]
	if name not in m:
		return
	for mapped_name in m[name]:
		map_name(m, line[1:], s + mapped_name + "\t")

with open(gene_name_mapping_file) as f:
	for line in f:
		line = [k.strip() for k in line.strip().split("\t")]
		if line[1-forward] not in m:
			m[line[1-forward]] = set()
		if forward < len(line):
			m[line[1-forward]].add(line[forward])


try:
	for line in sys.stdin:
		line = line.strip()
		line_split = [k.strip() for k in line.split("\t")]
		key = line_split[col]
		before_line = "\t".join(line_split[:col])
		after_line = "\t".join(line_split[col+1:])
		if key not in m:
			if print_mismatch: print line 
			continue
		for value in m[key]:
			s = before_line + "\t" + value + "\t" + after_line
			s = s.strip()
			print s
except IOError, ioe:
	pass
