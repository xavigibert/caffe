#!/bin/bash

TLIST="fastVsBg fastVsFast"
CLIST="00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15"

for TASK in $TLIST; do
echo ${TASK}
for CL in $CLIST; do

np=`cat ${TASK}_?_${CL}.csv | grep -c '^1,'`
nn=`cat ${TASK}_?_${CL}.csv | grep -c '^0,'`
echo "scale=5; (${np}-${nn})/(${np}+${nn})"|bc

done
done