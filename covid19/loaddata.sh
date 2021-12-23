#!/bin/bash

cdir=$(pwd)
if [ ! -d "/home/adivp416/public_html/covid19/github" ]; then
   git clone https://github.com/CSSEGISandData/COVID-19.git /home/adivp416/public_html/covid19/COVID-19
   ln -s /home/adivp416/public_html/covid19/COVID-19/csse_covid_19_data/csse_covid_19_time_series /home/adivp416/public_html/covid19/github
fi
cd /home/adivp416/public_html/covid19/github
git stash
git pull

sed -i 's/"Korea, South"/South Korea/g' time_series_covid19_confirmed_global.csv
sed -i 's/"Korea, South"/South Korea/g' time_series_covid19_deaths_global.csv
sed -i 's/Taiwan\*/Taiwan/g' time_series_covid19_confirmed_global.csv
sed -i 's/Taiwan\*/Taiwan/g' time_series_covid19_deaths_global.csv
rm *.csv.?
cd ../
wget -O toronto_cases.csv https://ckan0.cf.opendata.inter.prod-toronto.ca/datastore/dump/e5bf35bc-e681-43da-b2ce-0242d00922ad
wget -O ontario_cases.csv https://data.ontario.ca/dataset/1115d5fe-dd84-4c69-b5ed-05bf0c0a0ff9/resource/d1bfe1ad-6575-4352-8302-09ca81f7ddfc/download/cases_by_status_and_phu.csv
cd $cdir
