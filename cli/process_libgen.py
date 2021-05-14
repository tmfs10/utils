#!/usr/bin/env python

import sys
import numpy
import re

if len(sys.argv) < 3:
	print >> sys.stderr, "Usage:", sys.argv[0], "<topic # list> <book content list>"
	sys.exit(1)

topic_list = set()
with open(sys.argv[1]) as f:
	for line in f:
		line = [k.strip() for k in line.strip().split("\t")]
		topic_list.add(int(line[0]))

backup_map = {
		"bio": re.compile(r"\bbio"),
		"biochem": re.compile(r"\bbiochem"),
		"cell|molecule": re.compile(r"\bcell|\bmolecul"),
		"cancer": re.compile(r"\bcancer\b"),
		"inorganic|chem": re.compile(r"\binorganic\b|\bchem"),
		"chem": re.compile(r"\bchem"),
		"data mining": re.compile(r"\bdata mining\b"),
		"neural network": re.compile(r"\bneural network\b"),
		"biomed": re.compile(r"\bbiomed"),
		"mechanical|engineer": re.compile(r"\bmechanical\b|\bengineer"),
		"statistical": re.compile(r"\bstatistical\b"),
		"dynamical": re.compile(r"\bdynamical\b"),
		"dynamics": re.compile(r"\bdynamics\b"),
		"chaos": re.compile(r"\bchaos\b"),
		"equilibr": re.compile(r"\bequilibr"),
		"hamilton": re.compile(r"\bhamilton\b"),
		"gene": re.compile(r"\bgene\b"),
		"virus": re.compile(r"\bvirus\b"),
		"neural": re.compile(r"\bneural\b"),
		"microb": re.compile(r"\bmicrob"),
		"tnf": re.compile(r"\btnf\b"),
		"ecrf": re.compile(r"\becrf\b"),
		"bacteri": re.compile(r"\bbacteri"),
		}

with open(sys.argv[2]) as f:
	line_num = 0
	num_accepted = 0
	for line in f:
		line_num += 1
		line_split = [k.strip() for k in line.strip().split(",\"")]
		for j in reversed(xrange(len(line_split))):
			if "/" in line_split[j]:
				temp = line_split[j].split("/")
				if temp[0].isdigit() and temp[1][:32].isalnum():
					break
		if "/" not in line_split[j]:# str(j) + "\t" + str(line_num)
			md5_index = 34 if len(line_split[34]) == 33 else 35
			assert len(line_split[md5_index]) == 33, str(line_num)
			md5 = line_split[md5_index][:-1]
		else:
			md5 = line_split[j].split("/")[1][:32]
		topic = line_split[12]
		assert topic[-1] == "\"", str(line_num)
		assert len(md5) == 32, str(line_num)
		assert md5.isalnum(), str(line_num)
		
		book_name = line_split[1][:-1]
		authors = line_split[5][:-1]

		match_found = False
		for topic_name in backup_map:
			if backup_map[topic_name].search(line.lower()) != None:
				match_found = True
				print md5 + "\t\t" + book_name + " by " + authors + "\t" + topic_name
				num_accepted += 1
				break
		if match_found:
			continue

		topic_num = -1
		try:
			topic_num = int(topic[:-1])
		except ValueError, v:
			pass

		if topic_num in topic_list:
			num_accepted += 1
			print md5 + "\t\t" + book_name + " by " + authors + "\t" + str(topic_num)
		else:
			print >> sys.stderr, line_num, topic_num, book_name
	print >> sys.stderr, line_num
