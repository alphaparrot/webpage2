#!/bin/bash

cd /home/adivp416/public_html/
find /home/adivp416/public_html/covid19/ -type d -empty -print0 | xargs -0 -I {} /bin/rmdir "{}"
git stash
git pull
