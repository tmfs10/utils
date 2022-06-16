#!/usr/bin/env python

from __future__ import print_function
import sys
import argparse
import re
import os
import math
import random
import traceback
import gzip
import numpy as np
import pandas as pd

"""
first expression is begin and is run before the file loops
second expression is inner and is run inside the file loops
third expression is end and is run after the file loops

All ':' and indents have to be replaced by braces in the input code
All statements should end with a semi-colon
{} cannot be used for dicts currently, use dict() instead. Dict/Set comprehension not supported either.

break is replaced by raise BreakException
continue is replaced by raise ContinueException
TODO above two might be slow

TODO ';' cannot be put into a string right now. Fix that.

Special variables are:
	F[i][j] for jth column in ith file for a given line. Can only be used in inner expr. Indexed starting from 0
	lines[i] for the line in the ith file. Can only be used in inner expr. Indexed starting from 0.
	___f for filename
	___index for inner_expr_index
	NR for line number
"""

def make_list_float(l):
	return [float(k) for k in l]

def make_list_string(l, sep="\t"):
	return sep.join((str(k) for k in l))

def nested_add_to_dict(d, new_val_fn, *args):
	assert type(d) == dict
	for i in range(len(args)):
		a = args[i]
		t = d
		for j in range(i):
			t = t[args[j]]
		if a not in t:
			if i == len(args)-1:
				t[a] = new_val_fn()
			else:
				t[a] = {}

class BreakException(BaseException):
	pass

class ContinueException(BaseException):
	pass

def indent(s):
	open_braces = 0
	out_s = ""
	i = 0
	while i < len(s):
		c = s[i]
		if c == '{':
			open_braces += 1
			out_s += ':\n' + '\t'*open_braces
			i += 1
			while s[i] in [" ", "\t"]:
				i += 1
		elif c == '}':
			open_braces -= 1
			out_s += '\n#XXendXX\n' + '\t'*open_braces
			i += 1
			while i < len(s) and s[i] in [" ", "\t"]:
				i += 1
		elif c == '\n' or c == ";":
			out_s += '\n' + '\t'*open_braces
			i += 1
			while i < len(s) and s[i] in [" ", "\t"]:
				i += 1
		else:
			out_s += c
			i += 1

	out_s = out_s.split("\n")
	nesting = []
	for i in range(len(out_s)):
		line = out_s[i]
		if line.lstrip().startswith("for") and line.rstrip()[-1] == ":":
			nesting += ["for"]
			continue
		elif (line.lstrip().startswith("if") or line.lstrip().startswith("elif") or line.lstrip().startswith("else")) and line.rstrip()[-1] == ":":
			nesting += ["if"]
			continue

		if "#XXendXX" in line:
			assert len(nesting) > 0, "\n"+"\n".join(out_s)
			assert line == "#XXendXX", line
			if nesting[-1] == "if":
				out_s[i] = "#XXendXXif"
			else:
				out_s[i] = "#XXendXXfor"
			nesting = nesting[:-1]
	out_s = "\n".join(out_s)
	return out_s

def process_break_continue(s):
	lines = s.split("\n")
	nesting = []
	for i in range(len(lines)):
		line = lines[i]
		# rstrip important as we only want this replacement when it's a top-level break or continue
		if line.lstrip().startswith("for") and line.rstrip()[-1] == ":":
			nesting += ["for"]
		if "#XXendXXfor" in line:
			nesting = nesting[:-1]
		if "for" in nesting:
			continue
		assert len(nesting) == 0
		if "break" == line.strip():
			lines[i] = line.replace("break", "raise BreakException()")
		elif "continue" == line.strip():
			lines[i] = line.replace("continue", "raise ContinueException()")
	return "\n".join(lines)

if len(sys.argv) < 4:
	print("Usage:", sys.argv[0], "<math begin expr> <inner loop expr>... <end expr> <files>", file=sys.stderr)
	print("Included modules:\n", "\tsys\n", "\targparse\n", "\tre\n", "\tos\n", "\tmath\n", "\trandom\n", "\ttraceback\n", "\tgzip\n", "\tnumpy as np\n", file=sys.stderr)
	sys.exit(1)

____end_expr_index = 100
for ___i in reversed(range(3, len(sys.argv))):
	if not os.path.isfile(sys.argv[___i]):
		____end_expr_index = ___i
		break

assert ____end_expr_index < len(sys.argv), str(____end_expr_index)
if ____end_expr_index < 3:
	print("Error: need to give begin, atleast one inner, and an end expr", file=sys.stderr)
	print("Usage:", sys.argv[0], "<math begin expr> <inner loop expr>... <end expr> <files>", file=sys.stderr)
	sys.exit(1)

___begin_expr = indent(sys.argv[1])
___inner_expr_list = [process_break_continue(indent(sys.argv[i])) for i in range(2, ____end_expr_index)]
___end_expr = indent(sys.argv[____end_expr_index])
___all_expr_list = [___begin_expr] + ___inner_expr_list + [___end_expr]
___files = sys.argv[____end_expr_index+1:]
filenames = [os.path.basename(___f) for ___f in ___files]
___quiet = len(___files) == 0

if not ___quiet:
	print("BEGIN", file=sys.stderr)
	print(___begin_expr, file=sys.stderr)
	print("INNER", file=sys.stderr)
	for (___i,___inner_expr) in zip(range(len(___inner_expr_list)), ___inner_expr_list):
		print("---- inner_expr", ___i, " ----", file=sys.stderr)
		print(___inner_expr, file=sys.stderr)
	print("END", file=sys.stderr)
	print(___end_expr, file=sys.stderr)
	print("", file=sys.stderr)

regex = {
		'var': re.compile(r'\bF\[([0-9]{1,2})\]'),
		'lines': re.compile(r'\blines\[([0-9]{1,2})\]'),
		}

# Record which inner expr have variables from which ___files
___inner_expr_to_fid = [set() for ____i in range(len(___inner_expr_list))]
for ___i in range(len(___inner_expr_list)):
	___inner_expr = ___inner_expr_list[___i]
	for ___m in regex['var'].findall(___inner_expr):
		if not ___quiet:
			print("___i:", ___i, ", M:", ___m, file=sys.stderr)
		___fid = int(___m[0])
		___inner_expr_to_fid[___i].add(___fid)
	for ___m in regex['lines'].findall(___inner_expr):
		assert len(___m) == 1
		___fid = int(___m[0])
		___inner_expr_to_fid[___i].add(___fid)

___begin_expr = compile(___begin_expr, '<string>', 'exec')
___inner_expr_list = [compile(___inner_expr, '<string>', 'exec') for ___inner_expr in ___inner_expr_list]
___end_expr = compile(___end_expr, '<string>', 'exec')

# Run BEGIN expr
try:
	exec(___begin_expr)
except BaseException as ____e:
	print("Exception in ___begin_expr", file=sys.stderr)
	raise ____e

# Run INNER exprs
for ___index in range(len(___inner_expr_list)):
	___inner_expr = ___inner_expr_list[___index]
	___file_obj_list = []
	___num_files = 0
	if len(___files) == 0:
		___file_obj_list = [sys.stdin]
	for ___fid in range(len(___files)):
		if ___fid in ___inner_expr_to_fid[___index]:
			if ___files[___fid].endswith(".gz"):
				___file_obj_list += [gzip.open(___files[___fid])]
			else:
				___file_obj_list += [open(___files[___fid])]
			___num_files += 1
		else:
			___file_obj_list += [None]
	NR = 0
	#assert ___num_files > 0, str(___index) + "\t" + str(len(___inner_expr_list))
	while True:
		NR += 1
		if NR%1000000 == 0 and not ___quiet:
			print(NR/1000000, "million lines done", file=sys.stderr)

		lines = []
		___allNone = True
		for ___fid in range(len(___file_obj_list)):
			___f = ___file_obj_list[___fid]
			if ___f is None:
				lines += [None]
				continue
			line = ___f.readline()
			if line == "":
				___file_obj_list[___fid] = None
				lines += [None]
				continue
			___allNone = False
			lines += [line]
			assert lines[-1] != "", len(line)

		if ___allNone:
			break
		F = []
		for line in lines:
			if line is None:
				F += [None]
			else:
				F += [[___k.strip() for ___k in line.strip().split("\t")]]
        FF=F[0]
		try:
			exec(___inner_expr)
		except BreakException as ____e:
			break
		except ContinueException as ____e:
			continue
		except Exception as ____e:
			____cl, ____exc, ____tb = sys.exc_info()
			print("Exception in inner_expr " + str(___index) + " at NR=" + str(NR) + " and code line=" + str(traceback.extract_tb(____tb)[-1][1]), file=sys.stderr)
			raise ____e
	for ___f in ___file_obj_list:
		if ___f is not None:
			___f.close()

# Run END expr
try:
	exec(___end_expr)
except BaseException as ____e:
	print("Exception in end_expr", file=sys.stderr)
	raise ____e
