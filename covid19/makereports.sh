#!/bin/bash



cd /home/adivp416/public_html/covid19

git stash
git pull

python /home/adivp416/public_html/covid19/covid19report.py report
cat /home/adivp416/public_html/covid19/reportlog.txt
cd $cdir
