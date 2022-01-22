#!/bin/bash

cd /home/adivp416/public_html/covid19/

ionice -c 3 python /home/adivp416/public_html/covid19/covid19report.py dataset_world
rm /home/adivp416/public_html/covid19/adivparadise_covid19data.hdf5
mv /home/adivp416/public_html/covid19/tmp_adivparadise_covid19data.hdf5 /home/adivp416/public_html/covid19/adivparadise_covid19data.hdf5
rm /home/adivp416/public_html/covid19/adivparadise_covid19data_slim.hdf5
mv /home/adivp416/public_html/covid19/tmp_adivparadise_covid19data_slim.hdf5 /home/adivp416/public_html/covid19/adivparadise_covid19data_slim.hdf5
cat /home/adivp416/public_html/covid19/reportlog.txt

