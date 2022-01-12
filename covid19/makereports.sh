#!/bin/bash



cd /home/adivp416/public_html/covid19

ionice -c 3 git stash
ionice -c 2 -n 7 git pull

ionice -c 3 python /home/adivp416/public_html/covid19/covid19report.py report
cat /home/adivp416/public_html/covid19/reportlog.txt
cd $cdir
