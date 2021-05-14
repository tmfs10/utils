#!/bin/bash

if [ $# -lt 2 ]
then
	echo "Usage: $0 <i> <j> [<file>]"
	exit 1
fi

if [ $# -gt 2 ]
then
	pyexec.py "f=dict()" "k=F[0][0]; if k not in f { f[k] = [] } if len(f[k]) < $2 { f[k] += [F[0][1]] }" "for k in f { for l in f[k][$1-1:] { print k + '\t' + l } }" < $3
else
	pyexec.py "f=dict()" "k=F[0][0]; if k not in f { f[k] = [] } if len(f[k]) < $2 { f[k] += [F[0][1]] }" "for k in f { for l in f[k][$1-1:] { print k + '\t' + l } }"
fi
