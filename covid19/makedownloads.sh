#!/bin/bash

cdir=$(pwd)
if [ ! -d "/home/adivp416/public_html/covid19/github" ]; then
   mkdir /home/adivp416/public_html/covid19/github
fi
cd /home/adivp416/public_html/covid19/github

ionice -c 3 git stash
ionice -c 2 -n 7 git pull

ionice -c 2 -n 7 wget --limit-rate=2M -O /home/adivp416/public_html/covid19/github/time_series_covid19_confirmed_global.csv https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
ionice -c 2 -n 7 wget --limit-rate=2M -O /home/adivp416/public_html/covid19/github/time_series_covid19_confirmed_US.csv https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv
ionice -c 2 -n 7 wget --limit-rate=2M -O /home/adivp416/public_html/covid19/github/time_series_covid19_deaths_global.csv https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv
ionice -c 2 -n 7 wget --limit-rate=2M -O /home/adivp416/public_html/covid19/github/time_series_covid19_recovered_global.csv https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv
ionice -c 2 -n 7 wget --limit-rate=2M -O /home/adivp416/public_html/covid19/github/time_series_covid19_deaths_US.csv https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv

ionice -c 2 -n 7 sed -i 's/"Korea, South"/South Korea/g' /home/adivp416/public_html/covid19/github/time_series_covid19_confirmed_global.csv
ionice -c 2 -n 7 sed -i 's/"Korea, South"/South Korea/g' /home/adivp416/public_html/covid19/github/time_series_covid19_deaths_global.csv
ionice -c 2 -n 7 sed -i 's/Taiwan\*/Taiwan/g' /home/adivp416/public_html/covid19/github/time_series_covid19_confirmed_global.csv
ionice -c 2 -n 7 sed -i 's/Taiwan\*/Taiwan/g' /home/adivp416/public_html/covid19/github/time_series_covid19_deaths_global.csv
cd /home/adivp416/public_html/covid19/
ionice -c 2 -n 7 wget --limit-rate=2M -O /home/adivp416/public_html/covid19/toronto_cases.csv https://ckan0.cf.opendata.inter.prod-toronto.ca/datastore/dump/e5bf35bc-e681-43da-b2ce-0242d00922ad
ionice -c 2 -n 7 wget --limit-rate=2M -O /home/adivp416/public_html/covid19/ontario_cases.csv https://data.ontario.ca/dataset/1115d5fe-dd84-4c69-b5ed-05bf0c0a0ff9/resource/d1bfe1ad-6575-4352-8302-09ca81f7ddfc/download/cases_by_status_and_phu.csv

cd $cdir

cd /home/adivp416/public_html/covid19/
ionice -c 3 python /home/adivp416/public_html/covid19/covid19report.py datasets
cat /home/adivp416/public_html/covid19/reportlog.txt
