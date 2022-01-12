#!/bin/bash

cd /home/adivp416/public_html/covid19/
ionice -c 3 python /home/adivp416/public_html/covid19/covid19report.py makeshell
cat /home/adivp416/public_html/covid19/reportlog.txt
