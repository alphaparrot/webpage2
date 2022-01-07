

def dectime(timeseries,plot=True,baseline=7):
    #Return the time it takes to increase by a factor of 10, in days
    lastweek = timeseries[-baseline:]
    time = np.arange(baseline)+1.0
    m,b = np.polyfit((time),np.log10(lastweek),1)
    tfold = (1.0/m)
    if plot:
        plt.scatter(time,np.log10(lastweek),marker='.')
        plt.plot(time,np.log10(10**(m*(time)+b)))
        #plt.xscale('log')
        plt.ylabel("log$_{10}$ N")
        plt.xlabel("Time [days]")
        #plt.yscale('log')
    return tfold,m

def x2time(timeseries,plot=True,baseline=7):
    #Return the time it takes to increase by a factor of 2, in days
    lastweek = timeseries[-baseline:]
    time = np.arange(baseline)+1.0
    m,b = np.polyfit((time),np.log2(lastweek),1)
    tfold = (1.0/m)
    if plot:
        plt.scatter(time,np.log2(lastweek),marker='.')
        plt.plot(time,np.log2(2**(m*(time)+b)))
        #plt.xscale('log')
        plt.ylabel("log$_{2}$ N")
        plt.xlabel("Time [days]")
        #plt.yscale('log')
    return tfold,m

def day5avg(data):
    #now actually 7-day
    smooth = np.zeros(len(data)-6)
    #smooth[0] = np.nanmean(data[:4])
    #smooth[1] = np.nanmean(data[:5])
    #smooth[2] = np.nanmean(data[:6])
    for n in range(len(data)-6):
        smooth[n] = np.nanmean(data[n:n+7])
    #smooth[-3] = np.nanmean(data[-6:])
    #smooth[-2] = np.nanmean(data[-5:])
    #smooth[-1] = np.nanmean(data[-4:])
    return smooth

def week2avg(data,iscum=False):
    if iscum:
        y = np.diff(data)
    else:
        y = np.copy(data)
    smooth = np.zeros(len(y)-14)
    for n in range(len(smooth)):
        smooth[n] = np.nanmean(y[n:n+14])
    for n in range(13,0,-1):
        smooth = np.append(smooth,np.nanmean(y[-n:]))
    return smooth

def active3wk(data):
    cumsum = np.zeros(len(data)-1-21)
    for n in range(21,len(data)-1):
        cumsum[n-21] = np.sum(np.diff(data)[n-21:n])
    return cumsum

def active2wk(data):
    cumsum = np.zeros(len(data)-1-14)
    for n in range(14,len(data)-1):
        cumsum[n-14] = np.sum(np.diff(data)[n-14:n])
    return cumsum

def Rt(dailycases,interval=7,averaged=True,override=False): #Effective reproductive number, computed using the Bayesian method described in Yap & Yong (2020, https://doi.org/10.1101/2020.06.02.20120188)
    reff = np.geomspace(0.01,10.0,num=1000)
    gamma = 1.0/7.0 # reciprocal of the serial interval (the time between infection and secondary infection), using the 7-day estimate for sars-cov-2 in Yap & Yong (2020).
                    # We could technically treat this as a free parameter and iterate over it, but this number is consistent with a number of sources, so we will use it.
    loglikelihood = np.zeros((len(dailycases)-1,1000))
    logposterior = np.zeros((len(dailycases)-(interval+1),1000))
    estimate = np.zeros(logposterior.shape[0])
    for t in range(1,len(dailycases)):
        k = dailycases[t]
        km1 = dailycases[t-1]
        loglikelihood[t-1,:] = k*(np.log(km1)+gamma*(reff-1)) - km1*np.exp(gamma*(reff-1)) - loggamma(k+1)
    for t in range(interval,loglikelihood.shape[0]):
        logposterior[t-interval,:] = np.sum(loglikelihood[t-interval:t+1,:],axis=0)
    idx = np.argmax(logposterior,axis=1)
    if not averaged:
        dailycases = day5avg(dailycases)
    lim = 3.0
    if override:
        lim = 0.0
    for t in range(logposterior.shape[0]):
        estimate[t] = reff[idx[t]]*(dailycases[t-logposterior.shape[0]]>=0.0)                       
    return estimate,logposterior,loglikelihood

def matchtimes(tnew,tdata,data):
    newdata = np.zeros(len(tnew))
    olddata = {}
    for t in tnew:
        olddata[t] = 0.0
    for t in tdata:
        idx = np.argmin(abs(tdata-t))
        olddata[t] = data[idx]
    times = sorted(olddata.keys())
    for idx in range(len(times)):
        newdata[idx] = olddata[times[idx]]
    return newdata

def getcountries(dataset):
    countries = []
    for line in dataset:
        linedata = line.split(',')
        if len(linedata[0])>0:
            if linedata[0][0]=='"': #This was supposed to be one column
                place = (linedata[0]+','+linedata[1])[1:-1] #Recombine and strip off the quotation marks
                restofline = linedata[2:] #All the other columns
                linedata = [place,] + restofline
        if linedata[1][0]=='"': #South Korea
            linedata[1] = linedata[1][1:-1]
        countries.append(linedata[1])
    return sorted(list(set(countries)))

def extract_country(dataset,country,firstdatecol=5):
    '''Give dataset as list of lines, country name, and
    which column is the first date of data (starting from 1)'''
    countrydata = {}
    it0 = firstdatecol-1 #We start our indexing at 0
    lines = []
    for line in dataset:
        linedata = line.split(',')
        if linedata[-1] == '':
            linedata = linedata[:-1]
        if len(linedata[0])>0:
            if linedata[0][0]=='"': #This was supposed to be one column
                place = (linedata[0]+','+linedata[1])[1:-1] #Recombine and strip off the quotation marks
                restofline = linedata[2:] #All the other columns
                linedata = [place,] + restofline
        #print(linedata[0:2])
        if linedata[1] == country:
            lines.append(linedata)
    usa=False
    if country=="US":
        usa=True
    if not usa:
        if len(lines)>1:
            for locale in lines:
                localname = locale[0]
                #print(localname)
                countrydata[localname] = np.array(locale[it0:]).astype(int) #convert strings to integers and ignore latitude/longitude; we don't have half-people
            totals = np.zeros(countrydata[list(countrydata.keys())[0]].shape)
            for place in countrydata:
                totals += countrydata[place]
            countrydata["Total"] = totals
        else:
            countrydata = {"Total": np.array(lines[0][it0:]).astype(int)}
    else:
        statenames = {"Alabama"       :"AL",
                      "Arkansas"      :"AR",
                      "Alaska"        :"AK",
                      "Arizona"       :"AZ",
                      "California"    :"CA",
                      "Colorado"      :"CO",
                      "Connecticut"   :"CT",
                      "Delaware"      :"DE",
                      "Florida"       :"FL",
                      "Georgia"       :"GA",
                      "Hawaii"        :"HI",
                      "Idaho"         :"ID",
                      "Illinois"      :"IL",
                      "Indiana"       :"IN",
                      "Iowa"          :"IA",
                      "Kansas"        :"KS",
                      "Kentucky"      :"KY",
                      "Louisiana"     :"LA",
                      "Maine"         :"ME",
                      "Maryland"      :"MD",
                      "Massachusetts" :"MA",
                      "Michigan"      :"MI",
                      "Minnesota"     :"MN",
                      "Mississippi"   :"MS",
                      "Missouri"      :"MO",
                      "Montana"       :"MT",
                      "Nebraska"      :"NE",
                      "Nevada"        :"NV",
                      "New Hampshire" :"NH",
                      "New Jersey"    :"NJ",
                      "New Mexico"    :"NM",
                      "New York"      :"NY",
                      "North Carolina":"NC",
                      "North Dakota"  :"ND",
                      "Ohio"          :"OH",
                      "Oklahoma"      :"OK",
                      "Oregon"        :"OR",
                      "Pennsylvania"  :"PA",
                      "Rhode Island"  :"RI",
                      "South Carolina":"SC",
                      "South Dakota"  :"SD",
                      "Tennessee"     :"TN",
                      "Texas"         :"TX",
                      "Utah"          :"UT",
                      "Vermont"       :"VT",
                      "Virginia"      :"VA",
                      "Washington"    :"WA",
                      "West Virginia" :"WV",
                      "Wisconsin"     :"WI",
                      "Wyoming"       :"WY"}
        statecodes = {value: key for key, value in statenames.items()}
        ndates = len(lines[0][it0:])
        for state in statenames:
            countrydata[state] = np.zeros(ndates)
            
        for locale in lines:
            localname = locale[0]
            #print(localname)
            if localname not in statenames: 
                stateabbrev = localname.split(", ")[-1] #The abbreviation is always last, and it's always after a comma and a space
                #print(stateabbrev)
                if stateabbrev in statecodes: #We found a <city,state> pair
                    state = statecodes[stateabbrev]
                    #print(localname,state)
                    countrydata[state] += np.array(locale[it0:]).astype(int)
                    
                else: #It's DC or the Diamond Princess or something
                    if localname == "Jackson County, OR ":
                        state = statecodes["OR"]
                        countrydata[state] += np.array(locale[it0:]).astype(int)
                    elif localname != "Washington, D.C.":
                        countrydata[localname] = np.array(locale[it0:]).astype(int) 
                    
            else: #It's a state-level total
                state = localname
                countrydata[state] += np.array(locale[it0:]).astype(int)
                
        countrydata["Total"] = np.array(lines[0][it0:]).astype(int)#np.zeros(ndates)
        #for place in countrydata:
        #    countrydata["Total"] += countrydata[place]
    return countrydata

def extract_usa(dataset,firstdatecol=12):
    '''Give dataset as list of lines, country name, and
    which column is the first date of data (starting from 1)'''
    
    countrydata = {}
    states = []
    for line in dataset[1:]:
        states.append(line[6])
    states = list(set(states))
    for state in states:
        countrydata[state] = np.zeros(len(dataset[1])-firstdatecol+1)
        
    for line in dataset[1:]:
        state = line[6]
        countrydata[state] += np.array(line[firstdatecol-1:]).astype(float).astype(int)
        
    countrydata["Total"] = np.zeros(len(dataset[1])-firstdatecol+1)
    for state in states:
        countrydata["Total"] += countrydata[state]
        
    
    
    return countrydata

def extract_county(us_dataset,county,state="Minnesota",firstdatecol=12):
    '''Give dataset as list of lines, country name, and
    which column is the first date of data (starting from 1)'''

    found=False
    for line in us_dataset[1:]:
        stte = line[6]
        cty = line[5]
        if cty==county and stte==state:
            countydata = np.array(line[firstdatecol-1:]).astype(float).astype(int)
            return countydata
    print("%s not found!!"%county)
    raise Exception("We have a problem")

def get_counties(us_dataset,state="Minnesota"):
    '''Return list of counties for which data exists'''
    counties = {}
    found=False
    for line in us_dataset[1:]:
        stte = line[6]
        cty = line[5]
        if stte==state:
            counties[cty] = ""
    return sorted(counties.keys())

def get_countypop(county,state):
    county_csv = "co-est2019-alldata.csv"
    found=False
    with open(county_csv,"r") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            if row[5]==state:
                if county in row[6]:
                    return int(row[18])
    return -1

def plot_stateRt(state,dataset,timestamp):
    if not os.path.isdir(state):
        os.system('mkdir "%s"'%state)
    
    plt.axhline(1.0,linestyle='--',color='r')
    data = dataset[state]
    y = day5avg(np.diff(data))
    r,p,l = Rt(y,interval=7)
    week2r = week2avg(r)
    time1 = np.arange(len(r))-len(r)
    time2 = np.arange(len(week2r))-len(week2r)
    plt.plot(time2,week2r,marker='.',label="2-Week Average")
    plt.plot(time1,r,color='k',alpha=0.4,label="Instantaneous")
    plt.legend()
    plt.ylabel("$R_t$")
    plt.xlabel("Days Before Present")
    plt.title("%s Effective Reproductive Number"+timestamp)
    plt.savefig("%s/%s_rt.png"%(state,state),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_rt.pdf"%(state,state),bbox_inches='tight')
    plt.close('all')

def country_summary(country,dataset,deathdata,countrypops,timestamp):
    
    fig,axes = plt.subplots(4,1,sharex=True,figsize=(8,14))
    
    cdata = extract_country(dataset,country)
    y = cdata["Total"]
    y = day5avg(np.diff(y))
    r,p,l = Rt(y,interval=7)
    t1 = np.arange(len(y))-len(y)
    z = week2avg(r)
    dt = len(y)-len(z)
    t2 = np.arange(len(z))-len(z)
    t2r = np.arange(len(r))-len(r)
    axes[0].plot(t1,y/countrypops[country]*1e6,marker='',label=country)
    cfrac = cdata["Total"][-1]/countrypops[country]*1e2
    axes[0].annotate("%1.2f%% of people in %s have tested positive for COVID-19."%(cfrac,country),
                     (t1[0],y.max()/countrypops[country]))
    #axes[0].axvline((date(2020,12,9)-date.today()).days,color='g',linestyle='--')
    #axes[0].annotate("First Pfizer Shipment",((date(2020,12,9)-date.today()).days+5,100))
    ddata = extract_country(deathdata,country)
    y = ddata["Total"]
    y = day5avg(np.diff(y))/countrypops[country]*1e6
    t4 = np.arange(len(y))-len(y)
    axes[1].plot(t4,y,marker='',label=country)
    if ddata["Total"][-1]>=1:
        dfrac = int(round(countrypops[country]/ddata["Total"][-1]))
        axes[1].annotate("1 in %d people in %s have died from COVID-19."%(dfrac,country),
                         (t4[0],y.max()))
    axes[2].plot(t2,z,marker='.',label=country)
    axes[2].plot(t2r,r,marker='',color='k',alpha=0.4,label=country)
    axes[2].axhline(1.0,color='r',linestyle=':')
    
    y=cdata["Total"]/float(countrypops[country])*100
    t3 = np.arange(len(y))-len(y)
    axes[3].plot(t3,y,marker='',label=country)
    
    #axes[2].axvline((date(2020,12,9)-date.today()).days,color='g',linestyle='--')
    axes[3].set_xlabel("Time Before Present [days]")
    axes[2].set_ylabel("Effective Reproductive Number R$_t$")
    axes[1].set_ylabel("New Deaths per Day per 1M [7-day Average]")
    axes[3].set_ylabel("Cumulative Cases [percent of population]")
    axes[0].set_ylabel("New Cases per Day per 1M [7-day Average]")
    axes[0].set_yscale('log')
    axes[0].set_title(country+timestamp,size=14,fontweight='bold')
    plt.tight_layout()
    plt.savefig("%s_summary.png"%country,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_summary.pdf"%country,bbox_inches='tight')
    plt.close('all')

def plot_TOneighborhood(neighborhood,dataset,timestamp):
       
    fnamestub = neighborhood
    if "/" in fnamestub:
        fnamestub = fnamestub.replace("/","-")
       
    cases = dataset["units"][neighborhood]["CASES"]
    hospt = dataset["units"][neighborhood]["HOSPITALIZED"]
    fatal = dataset["units"][neighborhood]["FATAL"]
    recov = dataset["units"][neighborhood]["RECOVERED"]
    
    if not os.path.isdir("%s"%fnamestub):
        os.system('mkdir "%s"'%fnamestub)
    
    plt.plot(np.arange(len(cases))-len(cases),cases,marker='.')
    plt.annotate("Daily counts for the last 3 weeks:\n"+(", ".join(["%d",]*21))%tuple(cases[-21:]),
                 (-len(cases)+2,cases.max()*1.1))
    #print("Daily counts for the last 3 weeks:\n"+(", ".join(["%d",]*21))%tuple(cases[-21:]),
    #             (-len(cases),cases.max()*1.1))
    #plt.yscale('log')
    plt.title("Daily New Cases in %s"%neighborhood+timestamp)
    plt.xlabel("Days Before Present")
    plt.ylabel("Cases per Day")
    plt.ylim(0,cases.max()*1.3)
    plt.savefig("%s/%s_rawcases.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_rawcases.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    curve = day5avg(cases)
    plt.plot(np.arange(len(curve))-len(curve),curve,marker='.')
    #plt.yscale('log')
    plt.title("Average Daily Cases in %s"%neighborhood+timestamp)
    plt.xlabel("Days Before Present")
    plt.ylabel("7-day Average Cases per Day")
    plt.savefig("%s/%s_avgcases.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_avgcases.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    plt.plot(np.arange(len(hospt))-len(hospt),hospt,marker='.')
    #plt.yscale('log')
    plt.title("Daily Hospitalizations in %s"%neighborhood+timestamp)
    plt.xlabel("Days Before Present")
    plt.ylabel("Hospitalizations per Day")
    plt.savefig("%s/%s_rawhosp.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_rawhosp.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    plt.plot(np.arange(len(fatal))-len(fatal),fatal,marker='.')
    #plt.yscale('log')
    plt.title("Daily Deaths in %s"%neighborhood+timestamp)
    plt.xlabel("Days Before Present")
    plt.ylabel("Deaths per Day")
    plt.savefig("%s/%s_rawdeaths.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_rawdeaths.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    curve = active3wk(np.cumsum(cases))
    plt.plot(np.arange(len(curve))-len(curve),curve,marker='.')
    #plt.yscale('log')
    plt.title("3-Week Running Sum of Cases in %s"%neighborhood+timestamp)
    plt.xlabel("Days Before Present")
    plt.ylabel("3-Wk Running Sum")
    plt.savefig("%s/%s_3wk.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_3wk.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    curve1 = active3wk(np.cumsum(fatal))
    curve2 = active3wk(np.cumsum(recov))
    plt.fill_between(np.arange(len(curve2))-len(curve2),curve1+curve2,curve1,
                     color='g',alpha=0.3,label="Recovered")
    curve3 = active3wk(np.cumsum(cases))
    plt.fill_between(np.arange(len(curve3))-len(curve3),curve3,curve1+curve2,
                     color='C1',alpha=0.6,label="Active")
    curve4 = active3wk(np.cumsum(hospt))
    plt.fill_between(np.arange(len(curve4))-len(curve4),curve4,
                     edgecolor='r',hatch='////',label="Hospitalized")
    plt.fill_between(np.arange(len(curve1))-len(curve1),curve1,color='k',alpha=1.0,label="Dead")
    #plt.yscale('log')
    plt.legend(loc=2)
    plt.ylim(0,1.1*curve3.max())
    plt.xlim(-len(curve4),0)
    plt.ylabel("3-week Running Sum")
    plt.xlabel("Days Before Present")
    plt.title("%s Cases"%neighborhood+timestamp)
    plt.savefig("%s/%s_breakdown.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_breakdown.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    fig,axes=plt.subplots(figsize=(14,9))
    curve = day5avg(cases)
    plt.plot(np.arange(len(curve))-len(curve),curve/dataset["units"][neighborhood]["POP"] * 1e5,marker='.',
             label=neighborhood)
    curve2 = day5avg(dataset["ProvincialTO"]["CASES"]/2.732e6 * 1e5)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,linestyle='--',color='k',alpha=1.0,label="Toronto")
    plt.legend()
    plt.xlabel("Days Before Present")
    plt.ylabel("7-day Avg Daily Cases per 100k")
    #plt.yscale('log')
    #plt.ylim(0.1,200.0)
    plt.xlim(-len(curve)+150,0)
    plt.title("Average Daily Cases per Capita in %s"%neighborhood+timestamp)
    plt.savefig("%s/%s_relcases.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_relcases.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    fig,axes=plt.subplots(figsize=(14,9))
    rt14,post14,like14 = Rt(day5avg(cases),interval=7,override=True)
    curve = week2avg(rt14)
    plt.plot(np.arange(len(curve))-len(curve),curve,marker='.',color='C0',label=neighborhood)
    plt.plot(np.arange(len(rt14))-len(rt14),rt14,alpha=0.4,
             color='C0',label="%s Instantaneous R$_t$"%neighborhood)
    rt14,post14,like14 = Rt(day5avg(dataset["ProvincialTO"]["CASES"]),interval=7)
    curve2 = week2avg(rt14)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,color='k',alpha=0.7,label="Toronto")
    plt.plot(np.arange(len(rt14))-len(rt14),rt14,alpha=0.4,linestyle='--',color='k')
    plt.legend(loc=2)
    plt.xlabel("Days Before Present")
    plt.ylabel("2-wk Mean Effective Reproductive Number R$_t$")
    #plt.yscale('log')
    plt.xlim(-len(curve)+125,0)
    plt.annotate("R$_t$ is the average number of people an infected \nperson will infect over the course of the infection. \nR$_t$>1 means daily cases will increase.",
                ((-len(curve)+125)/2.0-0.15*(len(curve)-125),1.9))
    plt.axhline(1.0,color='k',linestyle=':')
    plt.title("%s Effective Reproductive Number"%neighborhood+timestamp)
    plt.savefig("%s/%s_Rt.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_Rt.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    curve = day5avg(cases)
    plt.plot(np.arange(len(curve))-len(curve),curve,marker='.')
    #plt.yscale('log')
    plt.title("Average Daily Cases in %s"%neighborhood+timestamp)
    plt.xlabel("Days Before Present")
    plt.ylabel("7-day Average Cases per Day")
    plt.yscale('log')
    plt.savefig("%s/%s_avgcases_log.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_avgcases_log.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    curve = active3wk(np.cumsum(cases))
    plt.plot(np.arange(len(curve))-len(curve),curve,marker='.')
    #plt.yscale('log')
    plt.title("3-Week Running Sum of Cases in %s"%neighborhood+timestamp)
    plt.xlabel("Days Before Present")
    plt.ylabel("3-Wk Running Sum")
    plt.yscale('log')
    plt.savefig("%s/%s_3wk_log.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_3wk_log.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    curve1 = active3wk(np.cumsum(fatal))
    curve2 = active3wk(np.cumsum(recov))
    plt.fill_between(np.arange(len(curve2))-len(curve2),curve1+curve2,curve1,
                     color='g',alpha=0.3,label="Recovered")
    curve3 = active3wk(np.cumsum(cases))
    plt.fill_between(np.arange(len(curve3))-len(curve3),curve3,curve1+curve2,
                     color='C1',alpha=0.6,label="Active")
    curve4 = active3wk(np.cumsum(hospt))
    plt.fill_between(np.arange(len(curve4))-len(curve4),curve4,
                     edgecolor='r',hatch='////',label="Hospitalized")
    plt.fill_between(np.arange(len(curve1))-len(curve1),curve1,color='k',alpha=1.0,label="Dead")
    plt.yscale('log')
    plt.legend(loc=2)
    plt.ylim(0,1.1*curve3.max())
    plt.xlim(-len(curve4),0)
    plt.ylabel("3-week Running Sum")
    plt.xlabel("Days Before Present")
    plt.title("%s Cases"%neighborhood+timestamp)
    plt.savefig("%s/%s_breakdown_log.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_breakdown_log.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')

    fig,axes=plt.subplots(figsize=(14,9))
    curve = day5avg(cases)
    plt.plot(np.arange(len(curve))-len(curve),curve/dataset["units"][neighborhood]["POP"] * 1e5,marker='.',
             label=neighborhood)
    curve2 = day5avg(dataset["ProvincialTO"]["CASES"]/2.732e6 * 1e5)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,linestyle='--',color='k',alpha=1.0,label="Toronto")
    plt.legend()
    plt.xlabel("Days Before Present")
    plt.ylabel("7-day Avg Daily Cases per 100k")
    plt.yscale('log')
    #plt.ylim(0.1,200.0)
    plt.xlim(-len(curve)+150,0)
    plt.title("Average Daily Cases per Capita in %s"%neighborhood+timestamp)
    plt.savefig("%s/%s_relcases_log.png"%(fnamestub,fnamestub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_relcases_log.pdf"%(fnamestub,fnamestub),bbox_inches='tight')
    plt.close('all')
    
def plotStateOrProvince(name,country,dataset,deaths_dataset,national_cases,national_deaths,population,timestamp):
    '''national_cases and national_deaths should be population-adjusted.'''
    
    fstub = name.replace(" ","_").replace("&","and")
    
    if not os.path.isdir(fstub):
        os.system("mkdir %s"%fstub)
    
    cumcases = dataset[name]
    cases = np.diff(np.append([0,],cumcases))
    
    national_dcases = np.diff(np.append([0,],national_cases))
    national_ddeaths = np.diff(np.append([0,],national_deaths))
    
    plt.plot(np.arange(len(cases))-len(cases),cases,label=name)
    plt.annotate("%d Raw\nCases per Day"%cases[-1],(-len(cases)*0.2,0.7*cases.max()))
    plt.annotate("%1.1f%% of the population in %s has tested positive."%(dataset[name][-1]/float(population)*1e2,name),(-len(cases),0.9*cases.max()))
    plt.xlabel("Days before Present")
    plt.ylabel("New Cases per Day")
    plt.title("%s Daily New Cases"+timestamp)
    plt.savefig("%s/%s_rawdaily.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_rawdaily.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    curve = day5avg(cases)
    curve2 = day5avg(national_dcases)
    plt.plot(np.arange(len(curve))-len(curve),curve,label=name)
    plt.annotate("%d Average\nCases per Day"%curve[-1],(-len(curve)*0.2,0.7*curve.max()))
    plt.xlabel("Days before Present")
    plt.ylabel("7-Day Average New Cases per Day")
    plt.title("%s Average Daily New Cases"%name+timestamp)
    plt.savefig("%s/%s_avgdaily.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_avgdaily.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    
    plt.plot(np.arange(len(cases))-len(cases),cases/float(population)*1e5,
             label=name)
    plt.plot(np.arange(len(national_dcases))-len(national_dcases),national_dcases*1e5,
             label=country,color='k',alpha=0.6)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("New Cases per 100k per Day")
    plt.title("%s Daily New Cases per 100k"%name+timestamp)
    plt.savefig("%s/%s_relrawdaily.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_relrawdaily.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    curve = day5avg(cases)/float(population)*1e5
    curve2 = day5avg(national_dcases)*1e5
    plt.plot(np.arange(len(curve))-len(curve),curve,label=name)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,color='k',
             alpha=0.6,label=country)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("7-Day Average New Cases per 100k per Day")
    plt.title("%s Average Daily New Cases per 100k"%name+timestamp)
    plt.savefig("%s/%s_relavgdaily.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_relavgdaily.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    plt.plot(np.arange(len(cases))-len(cases),cases/float(population)*1e5,
             label=name)
    plt.plot(np.arange(len(national_dcases))-len(national_dcases),national_dcases*1e5,
             label=country,color='k',alpha=0.6)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("New Cases per 100k per Day")
    plt.yscale('log')
    plt.title("%s Daily New Cases per 100k"%name+timestamp)
    plt.savefig("%s/%s_relrawdaily_log.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_relrawdaily_log.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    curve = day5avg(cases)/float(population)*1e5
    curve2 = day5avg(national_dcases)*1e5
    plt.plot(np.arange(len(curve))-len(curve),curve,label=name)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,color='k',
             alpha=0.6,label=country)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("7-Day Average New Cases per 100k per Day")
    plt.yscale('log')
    plt.title("%s Average Daily New Cases per 100k"%name+timestamp)
    plt.savefig("%s/%s_relavgdaily_log.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_relavgdaily_log.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    curve = day5avg(cases)
    r,p,l = Rt(curve)
    r2wk = week2avg(r)
    fig,ax = plt.subplots(figsize=(14,7))
    plt.plot(np.arange(len(r))-len(r),r,label="Instantaneous",color='k',alpha=0.4)
    plt.plot(np.arange(len(r2wk))-len(r2wk),r2wk,label="2-week Average",color='k')
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("Effective Reproductive Number R$_t$")
    plt.title("%s Effective Reproductive Number"%name+timestamp)
    plt.axhline(1.0,linestyle=':',color='r')
    plt.savefig("%s/%s_Rt.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_Rt.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    fig,ax = plt.subplots(figsize=(14,9))
    curve = np.diff(deaths_dataset[name][1:])
    plt.plot(np.arange(len(curve))-len(curve),curve,label=name)
    plt.annotate("%d Deaths per Day"%curve[-1],(-len(curve)*0.2,0.7*curve.max()))
    plt.xlabel("Days before Present")
    plt.ylabel("New Deaths per Day")
    plt.title("%s Daily Deaths"%name+timestamp)
    plt.savefig("%s/%s_rawdeaths.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_rawdeaths.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    
    fig,ax = plt.subplots(figsize=(14,9))
    curve = day5avg(np.diff(deaths_dataset[name][1:])/float(population)*1e6)
    ntldeaths = day5avg(national_ddeaths)*1e6
    plt.plot(np.arange(len(curve))-len(curve),curve,label=name)
    if np.cumsum(curve)[-1]>=1:
        plt.annotate("1 in %d people have died in %s, %s from COVID-19."%(int(round(population/float(deaths_dataset[name][-1]))),name,country),
                    (-len(curve),curve.max()))
    plt.plot(np.arange(len(ntldeaths))-len(ntldeaths),ntldeaths,color='k',alpha=0.4,label=country)
    plt.legend()
    plt.annotate("%1.2f Deaths per Day per 1M"%curve[-1],(-len(curve)*0.2,0.7*curve.max()))
    plt.xlabel("Days before Present")
    plt.ylabel("7-Day Average New Deaths per 1M per Day")
    plt.title("%s Average Daily Deaths per 1M"%name+timestamp)
    plt.savefig("%s/%s_relavgdeaths.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_relavgdeaths.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    
    curve = active3wk(cumcases)
    curve2 = active3wk(national_cases)*1000
    plt.plot(np.arange(len(curve))-len(curve),curve,label=name)
    plt.xlabel("Days before Present")
    plt.ylabel("3-week Running Sum of Cases per Day")
    plt.title("Recent COVID-19 Cases in %s"%name+timestamp)
    plt.savefig("%s/%s_3wk.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_3wk.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    plt.plot(np.arange(len(curve))-len(curve),curve,label=name)
    plt.xlabel("Days before Present")
    plt.ylabel("3-week Running Sum of Cases per Day")
    plt.title("Recent COVID-19 Cases in %s"%name+timestamp)
    plt.yscale('log')
    plt.savefig("%s/%s_3wk_log.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_3wk_log.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    curve = curve/float(population)*1000
    
    plt.plot(np.arange(len(curve))-len(curve),curve,label=name)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,color='k',alpha=0.4,label=country)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("3-week Running Sum of Cases per 1000 per Day")
    plt.title("Recent COVID-19 Cases in %s"%name+timestamp)
    plt.savefig("%s/%s_rel3wk.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_rel3wk.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    plt.plot(np.arange(len(curve))-len(curve),curve,label=name)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,color='k',alpha=0.4,label=country)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("3-week Running Sum of Cases per 1000 per Day")
    plt.title("Recent COVID-19 Cases in %s"%name+timestamp)
    plt.yscale('log')
    plt.savefig("%s/%s_rel3wk_log.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("%s/%s_rel3wk_log.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    
    
    
def plotOntario(phu,cases,deaths,active,recovered,timestamp,population=None):
    
    fstub = str.title(phu).replace('"','').replace('&','and').replace(",","").replace("/","_").replace(" ","_").replace("-","_")
    if not os.path.isdir("ontario_%s"%fstub):
        os.system("mkdir ontario_%s"%fstub)
    phuname = str.title(phu)
    
    plt.plot(np.arange(len(cases[phu]))-len(cases[phu]),cases[phu])
    plt.xlabel("Days before Present")
    plt.ylabel("New Cases per Day")
    plt.title("%s, ON Raw Cases per Day"%phuname+timestamp)
    if population is not None:
        plt.annotate("%1.2f%% of the population of %s, ON has been infected."%(np.cumsum(cases[phu])[-1]/population,phuname))
    plt.savefig("ontario_%s/%s_rawdaily.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_%s/%s_rawdaily.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    plt.plot(np.arange(len(cases[phu]))-len(cases[phu]),cases[phu])
    plt.xlabel("Days before Present")
    plt.ylabel("New Cases per Day")
    plt.title("%s, ON Raw Cases per Day"%phuname+timestamp)
    if population is not None:
        plt.annotate("%1.2f%% of the population of %s has been infected."%(np.cumsum(cases[phu])[-1]/population,phu))
    plt.yscale('log')
    plt.savefig("ontario_%s/%s_rawdaily_log.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_%s/%s_rawdaily_log.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    curve = day5avg(cases[phu])
    plt.plot(np.arange(len(curve))-len(curve),curve)
    plt.xlabel("Days before Present")
    plt.ylabel("Average New Cases per Day")
    plt.title("%s, ON Average Cases per Day"%phuname+timestamp)
    if population is not None:
        plt.annotate("%1.2f%% of the population of %s, ON has been infected."%(np.cumsum(cases[phu])[-1]/population,phuname))
    plt.savefig("ontario_%s/%s_avgdaily.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_%s/%s_avgdaily.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    plt.plot(np.arange(len(curve))-len(curve),curve)
    plt.xlabel("Days before Present")
    plt.ylabel("Average New Cases per Day")
    plt.title("%s, ON Average Cases per Day"%phuname+timestamp)
    if population is not None:
        plt.annotate("%1.2f%% of the population of %s has been infected."%(np.cumsum(cases[phu])[-1]/population,phu))
    plt.yscale('log')
    plt.savefig("ontario_%s/%s_avgdaily_log.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_%s/%s_avgdaily_log.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    plt.plot(np.arange(len(cases[phu]))-len(cases[phu]),active[phu])
    plt.xlabel("Days before Present")
    plt.ylabel("Active Cases")
    plt.title("%s, ON Daily Active Cases"%phuname+timestamp)
    plt.savefig("ontario_%s/%s_active.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_%s/%s_active.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    
    plt.plot(np.arange(len(cases[phu]))-len(cases[phu]),active[phu])
    plt.xlabel("Days before Present")
    plt.ylabel("Active Cases")
    plt.title("%s, ON Daily Active Cases"%phuname+timestamp)
    plt.yscale('log')
    plt.savefig("ontario_%s/%s_active_log.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_%s/%s_active_log.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    curve=day5avg(np.diff(deaths[phu]))
    plt.plot(np.arange(len(curve))-len(curve),curve)
    plt.xlabel("Days before Present")
    plt.ylabel("7-day Average Deaths per Day")
    plt.title("Average daily deaths from COVID-19 in %s, ON"%phuname+timestamp)
    plt.savefig("ontario_%s/%s_deaths.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_%s/%s_deaths.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
    fig,axes=plt.subplots(figsize=(14,9))
    rt,post14,like14 = Rt(day5avg(cases[phu]),interval=7,override=True)
    rt14 = week2avg(rt)
    plt.plot(np.arange(len(rt14))-len(rt14),rt14,color='k',label="2-Week Average")
    plt.plot(np.arange(len(rt))-len(rt),rt,color='k',linestyle='--',alpha=0.4,label="Instantaneous")
    plt.axhline(1.0,color='r',linestyle=':')
    plt.title("%s, ON Effective Reproductive Number"%phuname+timestamp)
    plt.xlabel("Days before Present")
    plt.ylabel("Effective Reproductive Number R$_t$")
    
    plt.savefig("ontario_%s/%s_Rt.png"%(fstub,fstub),bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_%s/%s_Rt.pdf"%(fstub,fstub),bbox_inches='tight')
    plt.close('all')
    
def plotCounty(county,state,countydataset,statedataset,statepopulation,timestamp):
    fstub1 = state.replace(" ","_").replace("&","and")
    fstub2 = str.title(county).replace('"','').replace('&','and').replace(",","").replace("/","_").replace(" ","_").replace("-","_")
    
    pathdir = "%s/%s"%(fstub1,fstub2)
    
    #print("%s_rawdaily.png"%pathdir)
    
    if not os.path.isdir(fstub1):
        os.system("mkdir %s"%fstub)
        
    #These are all cumulative
    cases = extract_county(countydataset,county,state=state)
    population = float(get_countypop(county,state))
    pathdir = "%s/%s"%(fstub1,fstub2)
    statecases = statedataset[state]
    
    dcases = np.diff(cases)
    dstatecases = np.diff(statecases)
    curve = dcases[:]
    plt.plot(np.arange(len(curve))-len(curve),curve,label=county)
    plt.annotate("%d Raw\nCases per Day"%curve[-1],(-len(curve)*0.2,0.7*curve.max()))
    plt.annotate("%1.1f%% of the population in %s County has tested positive."%(cases[-1]/float(population)*1e2,county),(-len(curve),0.9*curve.max()))
    plt.xlabel("Days before Present")
    plt.ylabel("New Cases per Day")
    plt.title("%s Daily New Cases"+timestamp)
    plt.savefig("%s_rawdaily.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_rawdaily.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
    curve = day5avg(np.diff(cases))
    curve2 = day5avg(np.diff(statecases))
    plt.plot(np.arange(len(curve))-len(curve),curve,label="%s County"%county)
    plt.annotate("%d Average\nCases per Day"%curve[-1],(-len(curve)*0.2,0.7*curve.max()))
    plt.xlabel("Days before Present")
    plt.ylabel("7-Day Average New Cases per Day")
    plt.title("%s County Average Daily New Cases"%county+timestamp)
    plt.savefig("%s_avgdaily.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_avgdaily.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
    
    plt.plot(np.arange(len(dcases))-len(dcases),dcases/float(population)*1e5,
             label="%s County"%county)
    plt.plot(np.arange(len(dstatecases))-len(dstatecases),dstatecases/float(statepopulation)*1e5,
             label=state,color='k',alpha=0.6)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("New Cases per 100k per Day")
    plt.title("%s County Daily New Cases per 100k"%county+timestamp)
    plt.savefig("%s_relrawdaily.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_relrawdaily.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
    curve = day5avg(dcases)/float(population)*1e5
    curve2 = day5avg(dstatecases)/float(statepopulation)*1e5
    plt.plot(np.arange(len(curve))-len(curve),curve,label="%s County"%county)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,color='k',
             alpha=0.6,label=state)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("7-Day Average New Cases per 100k per Day")
    plt.title("%s County Average Daily New Cases per 100k"%county+timestamp)
    plt.savefig("%s_relavgdaily.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_relavgdaily.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
    plt.plot(np.arange(len(dcases))-len(dcases),dcases/float(population)*1e5,
             label="%s County"%county)
    plt.plot(np.arange(len(dstatecases))-len(dstatecases),dstatecases/float(statepopulation)*1e5,
             label=state,color='k',alpha=0.6)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("New Cases per 100k per Day")
    plt.yscale('log')
    plt.title("%s County Daily New Cases per 100k"%county+timestamp)
    plt.savefig("%s_relrawdaily_log.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_relrawdaily_log.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
    curve = day5avg(dcases)/float(population)*1e5
    curve2 = day5avg(dstatecases)/float(statepopulation)*1e5
    plt.plot(np.arange(len(curve))-len(curve),curve,label="%s County"%county)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,color='k',
             alpha=0.6,label=state)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("7-Day Average New Cases per 100k per Day")
    plt.yscale('log')
    plt.title("%s County Average Daily New Cases per 100k"%county+timestamp)
    plt.savefig("%s_relavgdaily_log.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_relavgdaily_log.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
    curve = day5avg(dcases)
    r,p,l = Rt(curve)
    r2wk = week2avg(r)
    curve2 = day5avg(dstatecases)
    r2,p,l = Rt(curve2)
    r2wk2 = week2avg(r2)
    fig,ax = plt.subplots(figsize=(14,7))
    plt.plot(np.arange(len(r))-len(r),r,label="%s County R$_t$"%county,color='k',alpha=0.4)
    plt.plot(np.arange(len(r2wk))-len(r2wk),r2wk,label="%s County 2-week Average R$_t$"%county,color='k')
    plt.plot(np.arange(len(r2))-len(r2),r2,label="%s R$_t$"%state,color='C0',alpha=0.4)
    plt.plot(np.arange(len(r2wk2))-len(r2wk2),r2wk2,label="%s 2-week Average R$_t$"%state,color='C0')
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("Effective Reproductive Number R$_t$")
    plt.title("%s County Effective Reproductive Number"%county+timestamp)
    plt.axhline(1.0,linestyle=':',color='r')
    plt.savefig("%s_Rt.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_Rt.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
    curve = active3wk((cases))
    curve2 = active3wk((statecases))/float(statepopulation)*1000
    plt.plot(np.arange(len(curve))-len(curve),curve,label="%s County"%county)
    plt.xlabel("Days before Present")
    plt.ylabel("3-week Running Sum of Cases per Day")
    plt.title("Recent COVID-19 Cases in %s County"%county+timestamp)
    plt.savefig("%s_3wk.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_3wk.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
    plt.plot(np.arange(len(curve))-len(curve),curve,label="%s County"%county)
    plt.xlabel("Days before Present")
    plt.ylabel("3-week Running Sum of Cases per Day")
    plt.title("Recent COVID-19 Cases in %s County"%county+timestamp)
    plt.yscale('log')
    plt.savefig("%s_3wk_log.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_3wk_log.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
    curve = curve/float(population)*1000
    
    plt.plot(np.arange(len(curve))-len(curve),curve,label="%s County"%county)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,color='k',alpha=0.4,label=state)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("3-week Running Sum of Cases per 1000 per Day")
    plt.title("Recent COVID-19 Cases in %s County"%county+timestamp)
    plt.savefig("%s_rel3wk.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_rel3wk.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
    plt.plot(np.arange(len(curve))-len(curve),curve,label="%s County"%county)
    plt.plot(np.arange(len(curve2))-len(curve2),curve2,color='k',alpha=0.4,label=state)
    plt.legend()
    plt.xlabel("Days before Present")
    plt.ylabel("3-week Running Sum of Cases per 1000 per Day")
    plt.title("Recent COVID-19 Cases in %s County"%county+timestamp)
    plt.yscale('log')
    plt.savefig("%s_rel3wk_log.png"%pathdir,bbox_inches='tight',facecolor='white')
    plt.savefig("%s_rel3wk_log.pdf"%pathdir,bbox_inches='tight')
    plt.close('all')
    
      
    

def plotgroup(group,directory='mygroup'):
    if not os.path.isdir(directory):
        os.system("mkdir %s"%directory)
        
    curves = []
    for location in group:
        data = {}
        if location["type"]=="county":
            data["cases"] = extract_county(location["dataset"],location["place"],
                                           state=location["state/province"])
            data["population"] = float(get_countypop(location["place"],location["state/province"]))
            data["label"] = "%s, %s"%(location["place"],location["abbrev"])
        elif location["type"]=="state/province":
            data["cases"] = location["dataset"][location["place"]]
            data["population"] = float(location["population"])
            data["label"] = location["place"]
        elif location["type"]=="city":
            data["cases"] = location["dataset"]
            data["population"] = float(location["population"])
            data["label"] = "%s, %s"%(location["place"],location["abbrev"])
        elif location["type"]=="neighborhood":
            data["cases"] = np.cumsum(location["dataset"]["units"][location["place"]]["CASES"])
            data["population"] = float(location["dataset"]["units"][location["place"]]["POP"])
            data["label"] = "%s, %s"%(location["place"],location["abbrev"])
        data["time"] = np.arange(len(data["cases"]))-len(data["cases"])
        data["latest"] = location["timestamp"]
        curves.append(data)
      
    for place in curves:
        print("%s: %d cases; %d population; %f cases per 100"%(place["label"],place["cases"][-1],
                                                               place["population"],
                                                               place["cases"][-1]/float(place["population"])))
    
    fig,axes=plt.subplots(figsize=(14,10))
    for place in curves:
        plt.plot(place["time"],place["cases"]/float(place["population"])*1e2,
                 marker='.',label=place["label"])
        plt.annotate(place["label"],(1,place["cases"][-1]/float(place["population"])*1e2))
    #plt.legend()
    #plt.yscale('log')
    plt.ylabel("Total Confirmed Cases [% of Population]")
    plt.xlabel("Time Before Present [days]")
    #plt.xlim(40,len(hennepin))
    plt.title("COVID-19 in Family's Counties")
    plt.savefig("%s/fam_totalc_linpop.png"%directory,bbox_inches='tight',facecolor='white')
    plt.savefig("%s/fam_totalc_linpop.pdf"%directory,bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(14,10))
    for place in curves:
        avgcurve = day5avg(np.diff(place["cases"]))
        time = np.arange(len(avgcurve))-len(avgcurve)
        plt.plot(time,avgcurve/float(place["population"])*1e5,marker='.',label=place["label"])
        plt.annotate(place["label"],(1,avgcurve[-1]/float(place["population"])*1e5))
    plt.yscale('log')
    plt.ylabel("New Confirmed Cases per Day per 100k")
    plt.xlabel("Time Before Present [days]")
    #plt.xlim(40,len(hennepin))
    plt.ylim(0.01,2.0e2)
    plt.title("Family's Locations: New Cases per Day per 100k (7-day Average)")
    plt.savefig("%s/fam_newc_logpop.png"%directory,bbox_inches='tight',facecolor='white')
    plt.savefig("%s/fam_newc_logpop.pdf"%directory,bbox_inches='tight')
    plt.close('all')
    
    
    
    fig,ax=plt.subplots(figsize=(14,10))
    for place in curves:
        wk3curve = active3wk(place["cases"])
        time = np.arange(len(wk3curve))-len(wk3curve)
        plt.plot(time,wk3curve/float(place["population"])*1e5,marker='.',label=place["label"])
        plt.annotate(place["label"],(1,wk3curve[-1]/float(place["population"])*1e5))
    plt.ylabel("3-week Running Sum of Cases per 100k")
    plt.xlabel("Time Before Present [days]")
    #plt.xlim(40,len(hennepin)-21)
    plt.title("Family's Locations: 3-week Running Sum of Cases per 100k")
    plt.savefig("%s/fam_newc_linpop_3wk.png"%directory,bbox_inches='tight',facecolor='white')
    plt.savefig("%s/fam_newc_linpop_3wk.pdf"%directory,bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(14,10))
    for place in curves:
        wk3curve = active3wk(place["cases"])
        time = np.arange(len(wk3curve))-len(wk3curve)
        plt.plot(time,wk3curve/float(place["population"])*1e5,marker='.',label=place["label"])
        plt.annotate(place["label"],(1,wk3curve[-1]/float(place["population"])*1e5))
    plt.ylabel("3-week Running Sum of Cases per 100k")
    plt.xlabel("Time Before Present [days]")
    plt.yscale('log')
    #plt.xlim(40,len(hennepin)-21)
    plt.title("Family's Locations: 3-week Running Sum of Cases per 100k")
    plt.savefig("%s/fam_newc_logpop_3wk.png"%directory,bbox_inches='tight',facecolor='white')
    plt.savefig("%s/fam_newc_logpop_3wk.pdf"%directory,bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(14,8))
    for place in curves:
        r,p,l = Rt(day5avg(np.diff(place["cases"])))
        r = week2avg(r)
        time = np.arange(len(r))-len(r)
        plt.plot(time,r,marker='.',label=place["label"])
        plt.annotate(place["label"],(1,r[-1]))
    plt.ylabel("Effective Reproductive Number R$_t$")
    plt.xlabel("Time Before Present [days]")
    plt.axhline(1.0,linestyle=':',color='k')
    #plt.yscale('log')
    #plt.xlim(40,len(hennepin)-21)
    plt.title("Family's Locations: Effective Reproductive Number R$_t$")
    plt.savefig("%s/fam_rt.png"%directory,bbox_inches='tight',facecolor='white')
    plt.savefig("%s/fam_rt.pdf"%directory,bbox_inches='tight')
    plt.close('all')
    
 
def _log(destination,string):
    if destination is None:
        if string=="\n":
            print()
        else:
            print(string)
    else:
        with open(destination,"a") as f:
            f.write(string+"\n")   
    
def report():
    
    os.system('echo "Imports completed. \t%s'%systime.asctime(systime.localtime())+'">/home/adivp416/public_html/covid19/reportlog.txt')
    
    countrysetf = "countrypopulations.csv"
    with open(countrysetf,"r") as df:
        countryset = df.read().split('\n')[1:]
    if countryset[-1] == "":
        countryset = countryset[:-1]
    countrypops = {}
    for line in countryset:
        linedata = line.split(',')
        name = linedata[0]
        popx = float(linedata[4])
        countrypops[name] = popx
    
    canadasetf = "provincepopulations.csv"
    with open(canadasetf,"r") as df:
        canadaset = df.read().split('\n')[2:]
    if canadaset[-1] == "":
        canadaset = canadaset[:-1]
    provincepops = {}
    for line in canadaset:
        linedata = line.split(',')
        name = linedata[1]
        popx = float(linedata[-3])
        provincepops[name] = popx
        
    statesetf = "statepopulations.csv"
    with open(statesetf,"r") as df:
        stateset = df.read().split('\n')[2:]
    if stateset[-1] == "":
        stateset = stateset[:-1]
    statepops = {}
    for line in stateset:
        linedata = line.split(',')
        name = linedata[2]
        popx = float(linedata[3])
        statepops[name] = popx
        
    _log("/home/adivp416/public_html/covid19/reportlog.txt","Static CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
        
    globalconf = "github/time_series_covid19_confirmed_global.csv"

    with open(globalconf,"r") as df:
        dataset = df.read().split('\n')
    header = dataset[0].split(',')
    dataset = dataset[1:]
    if dataset[-1]=='':
        dataset = dataset[:-1]
    latestglobal = header[-1]
        
    countries = getcountries(dataset)
    

    usacsv = []
    with open("github/time_series_covid19_confirmed_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usacsv.append(row)
            
    usa = extract_usa(usacsv)

    canada = extract_country(dataset,"Canada")

    ddatasetf = "github/time_series_covid19_deaths_global.csv"

    with open(ddatasetf,"r") as df:
        ddataset = df.read().split('\n')
    header = ddataset[0].split(',')
    ddataset = ddataset[1:]
    if ddataset[-1]=='':
        ddataset = ddataset[:-1]
        
    usadcsv = []
    with open("github/time_series_covid19_deaths_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usadcsv.append(row)
            
    _log("/home/adivp416/public_html/covid19/reportlog.txt","Dynamic CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
    
    ca_deaths = extract_country(ddataset,"Canada")
    us_deaths = extract_usa(usadcsv)

    timestamps = []
    rtimes = []
    ftimes = []
    htimes = []
    TOneighborhoods = {"units":{}}
    with open("toronto_cases.csv","r") as df:
        torontocsv = df.read().split('\n')[1:]
        while torontocsv[-1]=="":
            torontocsv = torontocsv[:-1]
        torontoraw = np.zeros(len(torontocsv))
        for line in range(len(torontoraw)):
            timestamp = torontocsv[line].split(',')[9].split('-')
            timestamps.append(date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])))
            entry = torontocsv[line].split(',')
            if entry[11]=="ACTIVE":
                pass
            elif entry[11]=="FATAL":
                ftimes.append(timestamps[-1])
            else:
                rtimes.append(timestamps[-1])
            if entry[15]=="Yes":
                htimes.append(timestamps[-1])
                
            if entry[4] not in TOneighborhoods["units"]:
                TOneighborhoods["units"][entry[4]] = {"CASES":[],"FATAL":[],"HOSPITALIZED":[],"RECOVERED":[]}
            TOneighborhoods["units"][entry[4]]["CASES"].append(timestamps[-1])
            if entry[11]=="ACTIVE":
                pass
            elif entry[11]=="FATAL":
                TOneighborhoods["units"][entry[4]]["FATAL"].append(timestamps[-1])
            else:
                TOneighborhoods["units"][entry[4]]["RECOVERED"].append(timestamps[-1])
            if entry[15]=="Yes":
                TOneighborhoods["units"][entry[4]]["HOSPITALIZED"].append(timestamps[-1])
            
    #for neighborhood in TOneighborhoods["units"]:
    #    TOneighborhoods["units"][neighborhood]["CASES"] = np.array(TOneighborhoods["units"][neighborhood]["CASES"])
    #    TOneighborhoods["units"][neighborhood]["FATAL"] = np.array(TOneighborhoods["units"][neighborhood]["FATAL"])
    #    TOneighborhoods["units"][neighborhood]["HOSPITALIZED"] = np.array(TOneighborhoods["units"][neighborhood]["HOSPITALIZED"])
    #    TOneighborhoods["units"][neighborhood]["RECOVERED"] = np.array(TOneighborhoods["units"][neighborhood]["RECOVERED"])
            
    #All cases
    origin = np.array(timestamps).min()
    for line in range(len(timestamps)):
        day = ((timestamps[line]-origin).days)
        torontoraw[line] = int(day)
    #Recovered cases
    torontorraw = np.zeros(len(rtimes))
    for line in range(len(rtimes)):
        day = ((rtimes[line]-origin).days)
        torontorraw[line] = int(day)
    #Hospitalized cases
    torontohraw = np.zeros(len(htimes))
    for line in range(len(htimes)):
        day = ((htimes[line]-origin).days)
        torontohraw[line] = int(day)
    #Fatal cases
    torontofraw = np.zeros(len(ftimes))
    for line in range(len(ftimes)):
        day = ((ftimes[line]-origin).days)
        torontofraw[line] = int(day)
    
    timestamps = np.array(timestamps)
    
    times,toronto = np.unique(torontoraw,return_counts=True)
    rtimes,torontor = np.unique(torontorraw,return_counts=True)
    htimes,torontoh = np.unique(torontohraw,return_counts=True)
    ftimes,torontof = np.unique(torontofraw,return_counts=True)
    torontor = np.array(matchtimes(times,rtimes,torontor))
    torontoh = np.array(matchtimes(times,htimes,torontoh))
    torontof = np.array(matchtimes(times,ftimes,torontof))
    TOneighborhoods["casesTO"] = toronto
    TOneighborhoods["fatalTO"] = torontof
    TOneighborhoods["hosptTO"] = torontoh
    TOneighborhoods["recovTO"] = torontor
    for neighborhood in TOneighborhoods["units"]:
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["CASES"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["CASES"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["CASES"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["FATAL"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["FATAL"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["FATAL"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["HOSPITALIZED"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["HOSPITALIZED"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["HOSPITALIZED"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["RECOVERED"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["RECOVERED"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["RECOVERED"] = np.array(matchtimes(times,time,cases))

    popfile = open("neighbourhood-profiles-2016-csv.csv","r")
    popcsv = csv.reader(popfile,delimiter=',',quotechar='"',quoting=csv.QUOTE_ALL,skipinitialspace=True)
    n = 0
    for row in popcsv:
        if n==0:
            header = row[6:]
        elif n==3:
            poprow = row[6:]
            break
        n+=1
        
    neighborhoodpops = {}
    for n in range(len(header)):
        neighborhoodpops[header[n]] = float(poprow[n].replace(",",""))
    #neighborhoodpops.keys()

    if "" in TOneighborhoods["units"]:
        TOneighborhoods["units"].pop("")
        
    keys1 = sorted(neighborhoodpops.keys())
    keys2 = sorted(TOneighborhoods["units"].keys())
    #print(len(keys1),len(keys2))
    for n in range(len(keys1)):
        #print("%40s\t%s\t%s"%(keys1[n],str(keys1[n]==keys2[n]),keys2[n]))
        if keys1[n]!=keys2[n]:
            neighborhoodpops[keys2[n]] = neighborhoodpops[keys1[n]]
            neighborhoodpops.pop(keys1[n])
            
    for neighborhood in neighborhoodpops:
        TOneighborhoods["units"][neighborhood]["POP"] = neighborhoodpops[neighborhood]
            
    otimes = {}
    ontario = {}
    ontario_d = {}
    ontario_a = {}
    ontario_r = {}
    with open("ontario_cases.csv","r") as df:
        ontariocsv =df.read().split("\n")[1:]
        while ontariocsv[-1]=="":
            ontariocsv = ontariocsv[:-1]
    for line in ontariocsv:
        entry = line.split(',')
        if entry[1]=="":
            entry[1]="UNKNOWN"
        if entry[1][0]=='"':
            entry[1] = entry[1][1:]
        if entry[1][-1]=='"':
            entry[1] = entry[1][:-1]
        phu = entry[1]
        i0 = 0
        if entry[0][0]=='"':
            entry[0]=entry[0][1:]
        if entry[0][-1]=='"':
            entry[0]=entry[0][:-1]
        timestamp = entry[0].split('-')
        try:
            x = timestamp[1]
        except:
            timestamp = timestamp[0]
            timestamp = [timestamp[:4],timestamp[4:6],timestamp[6:]]
        if phu=='HALIBURTON' or phu=='KINGSTON':
            phu=entry[1]+','+entry[2]+','+entry[3]
            i0 = 2
        elif phu=='LEEDS':
            phu=entry[1]+','+entry[2]
            i0 = 1
        phu = phu.replace('"','')
        active = int(entry[3+i0])
        rec    = int(entry[4+i0])
        dead   = int(entry[5+i0])
        if phu not in otimes.keys():
            otimes[phu] = np.array([date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])),])
            ontario_a[phu] = np.array([active,])
            ontario_r[phu] = np.array([rec,])
            ontario_d[phu] = np.array([dead,])
        else:
            try:
                otimes[phu] = np.append(otimes[phu],date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])))
                ontario_a[phu] = np.append(ontario_a[phu],active)
                ontario_r[phu] = np.append(ontario_r[phu],rec)
                ontario_d[phu] = np.append(ontario_d[phu],dead)
            except:
                print(timestamp)
    for phu in otimes.keys():
        order = np.argsort(otimes[phu])
        otimes[phu] = otimes[phu][order]
        ontario_a[phu] = ontario_a[phu][order]
        ontario_r[phu] = ontario_r[phu][order]
        ontario_d[phu] = ontario_d[phu][order]
        ontario[phu] = (np.diff(np.append([0,],ontario_a[phu]))
                       +np.diff(np.append([0,],ontario_r[phu]))
                       +np.diff(np.append([0,],ontario_d[phu]))) 
                     #ontario_a[phu]+ontario_d[phu]+ontario_r[phu]

        
    latestON = otimes["TORONTO"].max() #datetime.date
    latestTO = timestamps.max()        #datetime.date
    latestusa = usacsv[0][-1]
    usatime = latestusa.split("/")
    latestusa = date(2000+int(usatime[2]),int(usatime[0]),int(usatime[1]))
    globaltime = latestglobal.split("/")
    latestglobal = date(2000+int(globaltime[2]),int(globaltime[0]),int(globaltime[1]))
    latestON = "\nAs of "+' '.join([x for i,x in enumerate(latestON.ctime().split()) if i!=3])
    latestTO = "\nAs of "+' '.join([x for i,x in enumerate(latestTO.ctime().split()) if i!=3])
    latestusa = "\nAs of "+' '.join([x for i,x in enumerate(latestusa.ctime().split()) if i!=3])
    latestglobal = "\nAs of "+' '.join([x for i,x in enumerate(latestglobal.ctime().split()) if i!=3])
        
    TOneighborhoods["ProvincialTO"] = {"CASES":ontario["TORONTO"],
                                       "FATAL":ontario_d["TORONTO"],
                                       "ACTIVE":ontario_a["TORONTO"],
                                       "RECOVERED":ontario_r["TORONTO"]}

    _log("/home/adivp416/public_html/covid19/reportlog.txt","Toronto data loaded. \t%s"%systime.asctime(systime.localtime()))

    #Generate HTML pages
    import makehtml
    for neighborhood in TOneighborhoods["units"]:
        fstub = neighborhood.replace("/","-")
        makehtml.makeneighborhood(neighborhood,"%s/%s"%(fstub,fstub))

    with open("index.html","r") as indexf:
        index = indexf.read().split('\n')
    html = []
    skipnext=False
    for line in index:
        if not skipnext:
            if "<!-- TORONTOFORM -->" in line:
                html.append(line)
                skipnext=True
                for neighborhood in sorted(TOneighborhoods["units"].keys()):
                    html.append('<!--TO-->\t\t\t<option value="%s">%s</option>'%(neighborhood,
                                                                          neighborhood))
            elif "<!-- PLACEHOLDER -->" in line:
                skipnext=True
            elif "<option" in line and "<!--TO-->" in line:
                pass #skip thi line too 
            else:
                html.append(line)
        else:
            skipnext=False
    with open("index.html","w") as indexf:
        indexf.write("\n".join(html))
        

    for k in sorted(ontario_a.keys()):
        fstub = str.title(k).replace('"','').replace('&','and').replace(",","").replace("/","_").replace(" ","_").replace("-","_")
        if os.path.isdir("ontario_%s"%fstub):
            makehtml.makephu(str.title(k),"ontario_%s/%s"%(fstub,fstub))
    
    with open("index.html","r") as indexf:
        index = indexf.read().split('\n')
    html = []
    skipnext=False
    for line in index:
        if not skipnext:
            if "<!-- ONTARIOFORM -->" in line:
                html.append(line)
                skipnext=True
                for k in sorted(ontario_a.keys()):
                    phu = "%s"%str.title(k)
                    html.append('<!--ON-->\t\t\t<option value="%s">%s</option>'%(phu,phu))
            elif "<!-- PLACEHOLDER -->" in line:
                skipnext=True
            elif "<option" in line and "<!--ON-->" in line:
                pass #skip thi line too 
            else:
                html.append(line)
        else:
            if "<!-- TRIPWIRE -->" in line:
                html.append(line)
            skipnext=False
    with open("index.html","w") as indexf:
        indexf.write("\n".join(html))
   
    cankeys = []
    for province in canada:
        if province!= "Total" and canada[province][-1]>150:
            cankeys.append(province)
   
    for k in sorted(cankeys):
        fstub = k.replace(" ","_").replace("&","and")
        makehtml.makeStateorProvince(k,"%s/%s"%(fstub,fstub))
   
    with open("index.html","r") as indexf:
        index = indexf.read().split('\n')
    html = []
    skipnext=False
    for line in index:
        if not skipnext:
            if "<!-- CANADAFORM -->" in line:
                html.append(line)
                skipnext=True
                for k in sorted(cankeys):
                    html.append('<!--CA-->\t\t\t<option value="%s">%s</option>'%(k,k))
            elif "<!-- PLACEHOLDER -->" in line:
                skipnext=True
            elif "<option" in line and "<!--CA-->" in line:
                pass #skip thi line too 
            else:
                html.append(line)
        else:
            if "<!-- TRIPWIRE -->" in line:
                html.append(line)
            skipnext=False
    with open("index.html","w") as indexf:
        indexf.write("\n".join(html))
        
    uskeys = []
    for state in usa:
        if us_deaths[state][-1]>=20 and state!="Total" and "Princess" not in state\
            and "Virgin Islands" not in state and "Military" not in state\
            and "Recovered" not in state and "Prisons" not in state\
            and "Hospitals" not in state and state != "Total" and usa[state][-1]>150:
            uskeys.append(state)
   
    for k in sorted(uskeys):
        fstub = k.replace(" ","_").replace("&","and")
        makehtml.makeStateorProvince(k,"%s/%s"%(fstub,fstub))
   
    #for state in uskeys:
        #counties = get_counties(usacsv,state=state)
        #for county in counties:
            #fstub1 = state.replace(" ","_").replace("&","and")
            #fstub2 = str.title(county).replace('"','').replace('&','and').replace(",","").replace("/","_").replace(" ","_").replace("-","_")
            #pathdir = "%s/%s"%(fstub1,fstub2)
            #makehtml.makeCounty(county,state,pathdir)
            
    ckeys = sorted(uskeys)
    ckeys.remove('Guam') #No counties in Guam
    
    with open("index.html","r") as indexf:
        index = indexf.read().split('\n')
    html = []
    skipnext=False
    for line in index:
        if not skipnext:
            if "<!-- USAFORM -->" in line:
                html.append(line)
                skipnext=True
                for k in sorted(uskeys):
                    html.append('<!--US-->\t\t\t<option value="%s">%s</option>'%(k,k))
            elif "<!-- COUNTYFORM -->" in line:
                html.append(line)
                skipnext=True
                n=0
                for k in sorted(ckeys):
                    html.append('<!--CY-->\t\t\t<option value="%d">%s</option>'%(n,k))
                    n+=1
            elif "<!-- PLACEHOLDER -->" in line:
                skipnext=True
            elif "<option" in line and ("<!--US-->" in line or "<!--CY-->" in line):
                pass #skip thi line too 
            else:
                html.append(line)
        else:
            if "<!-- TRIPWIRE -->" in line:
                html.append(line)
            skipnext=False
    with open("index.html","w") as indexf:
        indexf.write("\n".join(html))
        
    #Create linked state/county menus for the USA section
    with open("index.html","r") as indexf:
        index = indexf.read().split('\n')
    html = []
    for line in index:
        if "//COUNTY" in line:
            html.append(line)
            for state in ckeys:
                counties = get_counties(usacsv,state=state)
                options = '"'
                for county in counties:
                    options+="<option value='%s|%s'>%s</option>"%(county,state,county)
                options += '",'
                html.append("\t"+options+" //CY")
        elif "//CY" in line:
            pass
        else:
            html.append(line)
    with open("index.html","w") as indexf:
        indexf.write("\n".join(html))
    
    for country in sorted(countries):
        if os.path.exists("%s_summary.png"%country):
            makehtml.makeCountry(country)
    
    with open("index.html","r") as indexf:
        index = indexf.read().split('\n')
    html = []
    skipnext=False
    for line in index:
        if not skipnext:
            if "<!-- GLOBALFORM -->" in line:
                html.append(line)
                skipnext=True
                for country in sorted(countries):
                    if os.path.exists("%s_country.html"%(country.replace('"','').replace('&','and').replace(",","").replace("/","_").replace(" ","_").replace("-","_"))):
                        html.append('<!--GL-->\t\t\t<option value="%s">%s</option>'%(country,country))
            elif "<!-- PLACEHOLDER -->" in line:
                skipnext=True
            elif "<option" in line and "<!--GL-->" in line:
                pass #skip thi line too 
            else:
                html.append(line)
        else:
            if "<!-- TRIPWIRE -->" in line:
                html.append(line)
            skipnext=False
    with open("index.html","w") as indexf:
        indexf.write("\n".join(html))
        
        
    _log("/home/adivp416/public_html/covid19/reportlog.txt","Links and pages generated. \t%s"%systime.asctime(systime.localtime()))

    for neighborhood in TOneighborhoods["units"]:
        plot_TOneighborhood(neighborhood,TOneighborhoods,latestTO)
    _log("/home/adivp416/public_html/covid19/reportlog.txt","Toronto neighborhoods plotted. \t%s"%systime.asctime(systime.localtime()))
    
         
    fig,ax=plt.subplots(figsize=(16,12))
    for neighborhood in TOneighborhoods["units"]:
        curve = active3wk(np.cumsum(TOneighborhoods["units"][neighborhood]["CASES"]))/neighborhoodpops[neighborhood]*1e5
        plt.plot(range(len(curve)),curve,color='k',alpha=0.2)
        plt.annotate(neighborhood,(len(curve),curve[-1]),color='k',clip_on=True)
    #plt.yscale('log')
    #plt.ylim(10,500)
    plt.xlim(0,len(curve)*1.1)
    plt.xlabel("Days")
    plt.ylabel("3-week Running Sum of Cases per 100k")
    plt.title("Toronto Neighborhoods"+latestTO)
    plt.savefig("allneighborhoods.png",bbox_inches='tight',facecolor='white')
    plt.savefig("allneighborhoods.pdf",bbox_inches='tight')
    plt.close('all')
    
    torontopot = {}
    torontort = {}
    
    for neighborhood in TOneighborhoods["units"]:
        r,like,post = Rt(day5avg(TOneighborhoods["units"][neighborhood]["CASES"]),interval=7,override=True)
        torontort[neighborhood] = week2avg(r)
        z=active3wk(np.cumsum(TOneighborhoods["units"][neighborhood]["CASES"]))
        torontopot[neighborhood] = week2avg(z*r[-len(z):])
    
    fig,axes=plt.subplots(figsize=(16,9))
    xmin=0
    for neighborhood in TOneighborhoods["units"]:
        rt14,post14,like14 = Rt(day5avg(TOneighborhoods["units"][neighborhood]["CASES"]),interval=7,override=True)
        curve = week2avg(rt14)
        plt.plot(np.arange(len(curve))-len(curve),curve,marker='.',color='k',alpha=0.1)
        #plt.annotate(neighborhood,(0,curve[-1]))
        xmin=min(xmin,-len(curve))
    plt.xlabel("Days Before Present")
    plt.ylabel("2-wk Mean Effective Reproductive Number R$_t$")
    #plt.yscale('log')
    plt.xlim(xmin+125,0)
    #plt.ylim(0,3.0)
    plt.annotate("R$_t$ is the average number of people an infected \nperson will infect over the course of the infection. \nR$_t$>1 means daily cases will increase.",
                ((xmin+125)/2.0-0.15*(-xmin-125),2.5))
    plt.axhline(1.0,color='r',linestyle=':')
    plt.title("All Neighbourhood Reproductive Numbers"+latestTO)
    plt.savefig("neighbourhood_all_rt_zoom.png",bbox_inches='tight',facecolor='white')
    plt.savefig("neighbourhood_all_rt_zoom.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,axes=plt.subplots(figsize=(16,9))
    maxn = 0
    for neighborhood in TOneighborhoods["units"]:
        maxn = max(maxn,day5avg(TOneighborhoods["units"][neighborhood]["CASES"])[-1])
    
    xmin=0
    for neighborhood in TOneighborhoods["units"]:
        rt14,post14,like14 = Rt(day5avg(TOneighborhoods["units"][neighborhood]["CASES"]),interval=7,override=True)
        curve = week2avg(rt14)
        plt.plot(np.arange(len(curve))-len(curve),curve,marker='.',color='k',alpha=max(0.05,(day5avg(TOneighborhoods["units"][neighborhood]["CASES"])[-1]/float(maxn))**2))
        plt.annotate("%s (%1.2f/day)"%(neighborhood,day5avg(TOneighborhoods["units"][neighborhood]["CASES"])[-1]),(0,curve[-1]),alpha=max(0.05,(day5avg(TOneighborhoods["units"][neighborhood]["CASES"])[-1]/float(maxn))**2))
        xmin=min(xmin,-len(curve))
    plt.xlabel("Days Relative to Present")
    plt.ylabel("2-wk Mean Effective Reproductive Number R$_t$")
    #plt.yscale('log')
    plt.xlim(xmin+125,0)
    #plt.ylim(0,3.0)
    plt.annotate("R$_t$ is the average number of people an infected \nperson will infect over the course of the infection. \nR$_t$>1 means daily cases will increase.",
                ((xmin+125)/2.0-0.15*(-xmin-125),2.5))
    plt.axhline(1.0,color='r',linestyle=':')
    plt.title("All Neighbourhood Reproductive Numbers, Weighted by Current Daily Case Numbers"+latestTO)
    plt.savefig("neighbourhood_all_rt_zoom_weightdaily.png",bbox_inches='tight',facecolor='white')
    plt.savefig("neighbourhood_all_rt_zoom_weightdaily.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,axes=plt.subplots(figsize=(16,9))
    maxn = 0
    for neighborhood in TOneighborhoods["units"]:
        maxn = max(maxn,active3wk(np.cumsum(TOneighborhoods["units"][neighborhood]["CASES"]))[-1])
    
    xmin=0
    for neighborhood in TOneighborhoods["units"]:
        rt14,post14,like14 = Rt(day5avg(TOneighborhoods["units"][neighborhood]["CASES"]),interval=7,override=True)
        curve = week2avg(rt14)
        plt.plot(np.arange(len(curve))-len(curve),curve,marker='.',color='k',alpha=max(0.05,(active3wk(np.cumsum(TOneighborhoods["units"][neighborhood]["CASES"]))[-1]/float(maxn))**2))
        plt.annotate("%s (%d)"%(neighborhood,active3wk(np.cumsum(TOneighborhoods["units"][neighborhood]["CASES"]))[-1]),(0,curve[-1]),
                    alpha=max(0.05,(active3wk(np.cumsum(TOneighborhoods["units"][neighborhood]["CASES"]))[-1]/float(maxn))**2))
        xmin=min(xmin,-len(curve))
    plt.xlabel("Days Relative to Present")
    plt.ylabel("2-wk Mean Effective Reproductive Number R$_t$")
    #plt.yscale('log')
    plt.xlim(xmin+125,0)
    #plt.ylim(0,3.0)
    plt.annotate("R$_t$ is the average number of people an infected \nperson will infect over the course of the infection. \nR$_t$>1 means daily cases will increase.",
                ((xmin+125)/2.0-0.15*(-xmin-125),2.5))
    plt.axhline(1.0,color='r',linestyle=':')
    plt.title("All Neighbourhood Reproductive Numbers, Weighted by Active Cases"+latestTO)
    plt.savefig("neighbourhood_all_rt_zoom_weightactive.png",bbox_inches='tight',facecolor='white')
    plt.savefig("neighbourhood_all_rt_zoom_weightactive.pdf",bbox_inches='tight')
    plt.close('all')
    
    
    fig,ax=plt.subplots(figsize=(24,4))
    ptotals = {}
    for k in TOneighborhoods["units"]:
        rt14,post14,like14 = Rt(day5avg(TOneighborhoods["units"][k]["CASES"]),interval=7,override=True)
        ptotals[k] = week2avg(rt14)[-1]
    n=0
    labels = []
    growing = 0
    declining = 0
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        if ptotals[k]<1.0:
            declining+=1
        else:
            growing+=1
        if ptotals[k]>1.0:
            color='orange'
        elif ptotals[k]==1.0:
            color='blue'
        else:
            color='green'
        active=active3wk(np.cumsum(TOneighborhoods["units"][k]["CASES"]))[-1]
        bb=plt.bar(n,ptotals[k],alpha=max(0.05,(active3wk(np.cumsum(TOneighborhoods["units"][k]["CASES"]))[-1]/float(maxn))),color=color)
        bb=plt.bar(n,ptotals[k],alpha=1.0,facecolor='None',edgecolor='k')
            #ccl=bb.patches[0].get_facecolor()
            #plt.bar(n,ptotals[k],color=ccl,edgecolor='k')
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.axhline(2.9*0.3/.05,linestyle=':',color='brown')
    #ax.axhline(5.8*0.3/.05,linestyle=':',color='r')
    #plt.annotate("Current\nBeds",(n+2,4.0),xytext=(n+2,4.0))
    #plt.annotate("Surge\nBeds",(n+2,50.0),xytext=(n+2,50.0))
    #plt.annotate("This assumes hospitalization rates\nare 5% of confirmed cases,\nand assumes adequate testing\nsuch that confirmed cases are\nproportional to prevalence.\nIf testing is inadequate, the\ncritical care threshold will be reached\nmuch sooner. If we flatten\nthe curve, we may avert disaster.",
    #            (n+2,0.1),xytext=(n+2,0.1))
    #ax.set_ylim(1.0e-2,100)
    plt.axhline(1.0,linestyle='--',color='k')
    plt.xlim(-1,len(ptotals)+1)
    plt.ylabel("2-week Average R$_t$")
    
    offset = np.median(list(ptotals.values()))+0.5
    plt.annotate("Cases are increasing if R$_t>1$.",(0.4*len(ptotals),offset))
    plt.annotate("Increasing in %d neighborhoods\nDeclining in %d neighborhoods"%(growing,declining),(0.4*len(ptotals),offset-0.3))
    #plt.yscale('log')
    plt.title("Toronto Neighborhood Reproductive Numbers, Opacity Weighted by Active Cases"+latestTO)
    plt.savefig("neighborhoodRt_comparisonweighted.png",bbox_inches='tight',facecolor='white')
    plt.savefig("neighborhoodRt_comparisonweighted.pdf",bbox_inches='tight')
    plt.close('all')
    
    
    ptotals = {}
    for k in TOneighborhoods["units"]:
        rt14,post14,like14 = Rt(day5avg(TOneighborhoods["units"][k]["CASES"]),interval=7,override=True)
        ptotals[k] = week2avg(np.gradient(week2avg(rt14)))[-1]
    
    fig,ax=plt.subplots(figsize=(24,4))
    n=0
    labels = []
    previous=1.0
    change = -1.0
    growing=0
    declining=0
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        if ptotals[k]>0.0:
            color='orange'
        elif ptotals[k]==0.0:
            color='blue'
        else:
            color='green'
        
        if ptotals[k]>=0.0:
            growing+=1
        else:
            declining+=1
        active=active3wk(np.cumsum(TOneighborhoods["units"][k]["CASES"]))[-1]
        bb=plt.bar(n,ptotals[k],alpha=max(0.05,(active3wk(np.cumsum(TOneighborhoods["units"][k]["CASES"]))[-1]/float(maxn))),color=color)
        bb=plt.bar(n,ptotals[k],alpha=1.0,facecolor='None',edgecolor='k')
        if np.sign(previous)>0 and np.sign(ptotals[k])<0:
            change=n
        previous=ptotals[k]
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.axhline(2.9*0.3/.05,linestyle=':',color='brown')
    #ax.axhline(5.8*0.3/.05,linestyle=':',color='r')
    #plt.annotate("Current\nBeds",(n+2,4.0),xytext=(n+2,4.0))
    #plt.annotate("Surge\nBeds",(n+2,50.0),xytext=(n+2,50.0))
    #plt.annotate("This assumes hospitalization rates\nare 5% of confirmed cases,\nand assumes adequate testing\nsuch that confirmed cases are\nproportional to prevalence.\nIf testing is inadequate, the\ncritical care threshold will be reached\nmuch sooner. If we flatten\nthe curve, we may avert disaster.",
    #            (n+2,0.1),xytext=(n+2,0.1))
    #ax.set_ylim(1.0e-2,100)
    plt.axhline(0.0,linestyle='--',color='k')
    plt.xlim(-1,len(ptotals)+1)
    plt.axvline(0.5*(change+change-1),linestyle=':',color='r')
    plt.ylabel("2-week Average Derivative of <R$_t$>\n[Additional average infections per case per day]")
    plt.annotate("Transmission is increasing if $\delta$R$_t/\delta{t}>0$.",
                (0.51*len(ptotals),np.nanmax(list(ptotals.values()))*0.3))
    plt.annotate("Increasing in %d neighborhoods\nDeclining in %d neighborhoods"%(growing,declining),(0.51*len(ptotals),np.nanmax(list(ptotals.values()))*0.2))
    plt.annotate("Note: It is possible for transmission to increase while cases \n"+
                "are decreasing. What this means is people are getting less \n"+
                "cautious, and while on average sick people are infecting <1 \n"+
                "other people, they're passing the virus on to more and more \n"+
                "people, and eventually without changes in behavior, R$_t$ will\n"+
                "exceed 1 and cases will rise.",(0.1*len(ptotals),0.2*np.nanmax(list(ptotals.values()))))
    #plt.yscale('log')
    plt.title("Toronto Neighborhood Transmission, Opacity Weighted by Active Cases"+latestTO)
    plt.savefig("neighborhood_DRtdt_comparisonweighted.png",bbox_inches='tight',facecolor='white')
    plt.savefig("neighborhood_DRtdt_comparisonweighted.pdf",bbox_inches='tight')
    plt.close('all')
    
    plt.fill_between(times-times.max(),torontor+torontof,torontof,color='g',alpha=0.3,label='Recovered')
    plt.fill_between(times-times.max(),toronto,torontor+torontof,color='C1',alpha=0.6,label="Active")
    plt.fill_between(times-times.max(),torontoh,edgecolor='r',hatch='////',label="Hospitalized")
    plt.fill_between(times-times.max(),torontof,color='k',alpha=1,label='Dead')
    plt.yscale('symlog',linthreshy=100.0)
    plt.legend(loc=2)
    plt.ylim(0,1.1*toronto.max())
    plt.xlim(-times.max(),0)
    plt.ylabel("Cases per Day")
    plt.xlabel("Days Relative to Present")
    plt.title("Toronto Cases"+latestTO)
    plt.savefig("toronto_breakdown_log.png",bbox_inches='tight',facecolor='white')
    plt.savefig("toronto_breakdown_log.pdf",bbox_inches='tight')
    plt.close('all')
    
    plt.fill_between(times-times.max(),torontor+torontof,torontof,color='g',alpha=0.3,label='Recovered')
    plt.fill_between(times-times.max(),toronto,torontor+torontof,color='C1',alpha=0.6,label="Active")
    plt.fill_between(times-times.max(),torontoh,edgecolor='r',hatch='////',label="Hospitalized")
    plt.fill_between(times-times.max(),torontof,color='k',alpha=1,label='Dead')
    #plt.yscale('symlog',linthreshy=100.0)
    plt.legend(loc=2)
    plt.ylim(0,1.1*toronto.max())
    plt.xlim(-times.max(),0)
    plt.ylabel("Cases per Day")
    plt.xlabel("Days Relative to Present")
    plt.title("Toronto Cases"+latestTO)
    plt.savefig("toronto_breakdown.png",bbox_inches='tight',facecolor='white')
    plt.savefig("toronto_breakdown.pdf",bbox_inches='tight')
    plt.close('all')
    
    curve1 = active3wk(np.cumsum(torontof))
    curve2 = active3wk(np.cumsum(torontor))
    plt.fill_between(np.arange(len(curve2))-len(curve2),curve1+curve2,curve1,
                     color='g',alpha=0.3,label="Recovered")
    curve3 = active3wk(np.cumsum(toronto))
    plt.fill_between(np.arange(len(curve3))-len(curve3),curve3,curve1+curve2,
                     color='C1',alpha=0.6,label="Active")
    curve4 = active3wk(np.cumsum(torontoh))
    plt.fill_between(np.arange(len(curve4))-len(curve4),curve4,edgecolor='r',hatch='////',label="Hospitalized")
    plt.fill_between(np.arange(len(curve1))-len(curve1),curve1,color='k',alpha=1.0,label="Dead")
    plt.yscale('symlog',linthreshy=100.0)
    plt.legend(loc=2)
    plt.ylim(0,1.1*curve3.max())
    plt.xlim(-len(curve4),0)
    plt.ylabel("3-week Running Sum of Daily Cases")
    plt.xlabel("Days Relative to Present")
    plt.title("Toronto Cases"+latestTO)
    plt.savefig("toronto_breakdown_3wk_log.png",bbox_inches='tight',facecolor='white')
    plt.savefig("toronto_breakdown_3wk_log.pdf",bbox_inches='tight')
    plt.close('all')

    torontott = np.cumsum(ontario["TORONTO"][:])
    toronto = ontario["TORONTO"][:]
    torontopop = 2.732e6


    fig,axes=plt.subplots(figsize=(14,7))
    
    plt.axhline(1.0,linestyle='--',color='k')
    y = day5avg(toronto[:])
    r,p,l = Rt(y,interval=7)
    curve = week2avg(r)
    plt.plot(np.arange(len(curve))-len(curve),week2avg(r),marker='.',color='k',label="Toronto R$_t$ (2-wk Avg)")
    plt.plot(np.arange(len(r))-len(r),r,color='k',alpha=0.4,label="Toronto R$_t$")
    
    plt.ylabel("$R_t$ (Average People Each Sick Person Infects)")
    plt.xlabel("Days Before Present")
    plt.legend()
    plt.title("Toronto Effective Reproductive Number R$_t$"+latestON)
    plt.savefig("toronto_rt.png",bbox_inches='tight',facecolor='white')
    plt.savefig("toronto_rt.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(14,10))
    curve = day5avg(toronto[:])
    plt.plot(np.arange(len(curve))+1-len(curve),curve,marker='.')
    #plt.yscale('symlog',linthreshy=10.0)
    plt.ylim(0,toronto.max())
    plt.xlabel("Days Relative to Present",size=20)
    plt.ylabel("7-Day Average Cases per Day",size=20)
    plt.title("Toronto"+latestON,size=24)
    plt.annotate("%1.1f Average \nCases per Day"%curve[-1],(-150,800),size=24)
    plt.savefig("toronto_update.pdf",bbox_inches='tight')
    plt.savefig("toronto_update.png",bbox_inches='tight',facecolor='white')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(14,10))
    curve = toronto[:]
    plt.plot(np.arange(len(curve))+1-len(curve),curve,marker='.')
    #plt.yscale('symlog',linthreshy=10.0)
    plt.ylim(0,toronto.max())
    plt.xlabel("Days Relative to Present",size=20)
    plt.ylabel("Raw Cases per Day",size=20)
    plt.title("Toronto"+latestON,size=24)
    plt.annotate("%1.1f Raw\nCases per Day"%curve[-1],(-150,800),size=24)
    plt.savefig("toronto_update_raw.pdf",bbox_inches='tight')
    plt.savefig("toronto_update_raw.png",bbox_inches='tight',facecolor='white')
    plt.close('all')
    
    
    _log("/home/adivp416/public_html/covid19/reportlog.txt","Toronto plots completed. \t%s"%systime.asctime(systime.localtime()))
         
    fig,ax=plt.subplots(figsize=(12,10))
    for k in sorted(ontario_a.keys()):
        plt.plot(otimes[k],ontario_a[k])
        plt.annotate(str.title(k),(otimes[k][-1],ontario_a[k][-1]))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.yscale('log')
    plt.title("Ontario Active Cases"+latestON)
    plt.savefig("ontario_active.png",bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_active.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(12,10))
    for k in sorted(ontario.keys()):
        try:
            y = day5avg(ontario[k])
            plt.plot(np.arange(len(y))-len(y),y,label=k)
            plt.annotate(str.title(k),(0,y[-1]))
        except:
            pass
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.yscale('log')
    plt.xlabel("Days before Present")
    plt.title("Ontario New Cases [7-day average]"+latestON)
    plt.savefig("ontario_newcases.png",bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_newcases.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(12,10))
    for k in sorted(ontario_a.keys()):
        try:
            y = day5avg(np.diff(ontario_d[k]))
            plt.plot(np.arange(len(y))-len(y),y,label=k)
            plt.annotate(str.title(k),(0,y[-1]))
        except:
            print("Encountered an error with %s"%k)
            pass
    print("Passed the place where errors get thrown....")
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.yscale('log')
    plt.xlabel("Days before Present")
    plt.title("Ontario Daily Deaths [7-day average]"+latestON)
    plt.savefig("ontario_newdeaths.png",bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_newdeaths.pdf",bbox_inches='tight')
    plt.close('all')
    
    ptotals = {}
    maxn = 0
    fig,ax = plt.subplots(figsize=(14,9))
    for k in sorted(ontario_a.keys()):
        try:
            maxn = max(maxn,ontario_a[k][-1])
            r,p,l = Rt(day5avg(ontario[k]))
            r2wk = week2avg(r)
            ptotals[k] = r2wk[-1]
            plt.plot(np.arange(len(r2wk))-len(r2wk),r2wk,color='k',alpha=0.4)
            plt.annotate("%s"%str.title(k),(0,r2wk[-1]))
        except:
            pass
    plt.xlabel("Days before Present")
    plt.ylabel("2-week Average Effective Reproductive Number R$_tR")
    plt.title("Effective Reproductive Numbers of Ontario PHUs"+latestON)
    plt.axhline(1.0,color='r',linestyle=':')
    plt.xlim(-len(r2wk),0.2*len(r2wk))
    plt.savefig("ontario_allRt.png",bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_allRt.pdf",bbox_inches='tight')
    plt.close('all')
    
    
    fig,ax = plt.subplots(figsize=(15,4))
    n=0
    growing=0
    declining=0
    labels=[]
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(str.title(k))
        if ptotals[k]<1.0:
            declining+=1
        else:
            growing+=1
        if ptotals[k]>1.0:
            color='orange'
        elif ptotals[k]==1.0:
            color='blue'
        else:
            color='green'
        active = ontario_a[k][-1]
        bb = plt.bar(n,ptotals[k],alpha=max(0.15,.15+0.85*active/float(maxn)),edgecolor='k',color=color)
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    plt.axhline(1.0,linestyle='--',color='k')
    plt.ylabel("2-week Average R$_t$")
    plt.title("Ontario PHU Reproductive Numbers, Opacity Weighted by Active Cases"+latestON)
    plt.savefig("ontario_phuRtcomparison.png",bbox_inches='tight',facecolor='white')
    plt.savefig("ontario_phuRtcomparison.pdf",bbox_inches='tight')
    plt.close("all")
        
    
    ontariokeys = sorted(ontario_a.keys())
    for k in sorted(ontario_a.keys()):
        if len(ontario[k])>14:
            try:
                plotOntario(k,ontario,ontario_d,ontario_a,ontario_r,latestON)
            except:
                traceback.print_exc()
                os.system("rm -rf ontario_%s*"%(str.title(k).replace('"','').replace('&','and').replace(",","").replace("/","_").replace(" ","_").replace("-","_")))
                ontariokeys.remove(k)
        else:
            ontariokeys.remove(k)
    
    with open("index.html","r") as indexf:
        index = indexf.read().split('\n')
    html = []
    skipnext=False
    for line in index:
        if not skipnext:
            if "<!-- ONTARIOFORM -->" in line:
                html.append(line)
                skipnext=True
                for k in ontariokeys:
                    phu = "%s"%str.title(k)
                    html.append('<!--ON-->\t\t\t<option value="%s">%s</option>'%(phu,phu))
            elif "<!-- PLACEHOLDER -->" in line:
                skipnext=True
            elif "<option" in line and "<!--ON-->" in line:
                pass #skip thi line too 
            else:
                html.append(line)
        else:
            if "<!-- TRIPWIRE -->" in line:
                html.append(line)
            skipnext=False
    with open("index.html","w") as indexf:
        indexf.write("\n".join(html))
    _log("/home/adivp416/public_html/covid19/reportlog.txt","Ontario PHU plots completed. \t%s"%systime.asctime(systime.localtime()))
    
    cankeys = []
    for province in canada:
        if province!= "Total" and canada[province][-1]>150:
            plotStateOrProvince(province,"Canada",canada,ca_deaths,
                                extract_country(dataset,"Canada")["Total"]/float(countrypops["Canada"]),
                                extract_country(ddataset,"Canada")["Total"]/float(countrypops["Canada"]),
                                provincepops[province],latestglobal)
            cankeys.append(province)

    _log("/home/adivp416/public_html/covid19/reportlog.txt","Provincial plots completed. \t%s"%systime.asctime(systime.localtime()))
         
    adivgroup = [{"type":"county","dataset":usacsv,"state/province":"Minnesota",
                  "abbrev":"MN","place":"Hennepin","timestamp":latestusa},
                 {"type":"county","dataset":usacsv,"state/province":"Minnesota",
                  "abbrev":"MN","place":"Ramsey","timestamp":latestusa},
                 {"type":"county","dataset":usacsv,"state/province":"Minnesota",
                  "abbrev":"MN","place":"Dakota","timestamp":latestusa},
                 {"type":"county","dataset":usacsv,"state/province":"Oregon",
                  "abbrev":"OR","place":"Polk","timestamp":latestusa},
                 {"type":"county","dataset":usacsv,"state/province":"Oregon",
                  "abbrev":"OR","place":"Benton","timestamp":latestusa},
                 {"type":"county","dataset":usacsv,"state/province":"Oregon",
                  "abbrev":"OR","place":"Marion","timestamp":latestusa},
                 {"type":"county","dataset":usacsv,"state/province":"Florida",
                  "abbrev":"FL","place":"Pinellas","timestamp":latestusa},
                 {"type":"county","dataset":usacsv,"state/province":"Minnesota",
                  "abbrev":"MN","place":"Blue Earth","timestamp":latestusa},
                 {"type":"state/province","dataset":usa,"place":"Minnesota",
                  "population":statepops["Minnesota"],"timestamp":latestusa},
                 {"type":"city","place":"Toronto","abbrev":"ON","dataset":torontott,"population":torontopop,
                  "timestamp":latestON},
                 {"type":"state/province","dataset":canada,"place":"Newfoundland and Labrador",
                  "population":provincepops["Newfoundland and Labrador"],"timestamp":latestglobal},
                 {"type":"neighborhood","dataset":TOneighborhoods,"place":"The Beaches","abbrev":"Toronto",
                  "timestamp":latestTO}]
                 
    plotgroup(adivgroup,directory="adiv")

    _log("/home/adivp416/public_html/covid19/reportlog.txt","Adiv's personal report plotted. \t%s"%systime.asctime(systime.localtime()))
         
    fig,axes=plt.subplots(figsize=(14,10))
    for province in canada:
        place=province
        if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total":
            y = canada[province]/float(provincepops[province])*1e5
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y)-7)-(len(y)-6),day5avg(np.diff(y)),marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,day5avg(np.diff(y))[-1]))
    #plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1.0e-2,1000)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("New Cases per 100k")
    plt.title("Canada Daily Cases"+latestglobal)
    plt.savefig("ca_dailycasespop.png",bbox_inches='tight',facecolor='white')
    plt.savefig("ca_dailycasespop.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(14,14))
    for province in canada:
        place=province
        if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total": #Only plot places with >20 confirmed cases
            plt.plot(canada[place]/float(provincepops[place])*1e2,marker='.',label=place)
            coords = (len(canada[place]),canada[place][-1]/float(provincepops[place])*1e2)
            plt.annotate(place,coords,xytext=coords)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlim(30,len(canada["Total"]))
    #plt.ylim(0.1,30.0)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("Time [days]")
    plt.ylabel("Confirmed Cases (percentage of population)")
    plt.title("Canada Confirmed Cases"+latestglobal)
    plt.savefig("caconfirmed.png",bbox_inches='tight',facecolor='white')
    plt.savefig("caconfirmed.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,axes=plt.subplots(figsize=(14,10))
    for province in canada:
        place=province
        if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total":
            y = canada[province]/provincepops[province]*1e5
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y)-7)-(len(y)-6),day5avg(np.diff(y)),marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,day5avg(np.diff(y))[-1]))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(1.0e-2,100)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("New Cases per 100k")
    plt.title("Canada Daily Cases"+latestglobal)
    plt.savefig("canada_dailycasespop.png",bbox_inches='tight',facecolor='white')
    plt.savefig("canada_dailycasespop.pdf",bbox_inches='tight')
    plt.close('all')
    
    
    fig,axes=plt.subplots(figsize=(14,10))
    for province in canada:
        place=province
        if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total":
            y = canada[province]/provincepops[province]*1e5
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y)-7)-(len(y)-6),day5avg(np.diff(y)),marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,day5avg(np.diff(y))[-1]))
    #plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(1.0e-2,100)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("New Cases per 100k")
    plt.title("Canada Daily Cases"+latestglobal)
    plt.savefig("canada_dailycasespop_log.png",bbox_inches='tight',facecolor='white')
    plt.savefig("canada_dailycasespop_log.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,axes=plt.subplots(figsize=(14,10))
    for province in canada:
        place=province
        if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total":
            y = active3wk(canada[province])/provincepops[province]*1e2
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y))-len(y),y,marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,y[-1]))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(1.0,2.0e3)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("3-wk Cumulative Cases [Percent of Population]")
    plt.title("Canada Active Cases"+latestglobal)
    plt.savefig("canada_3wkcasespop.png",bbox_inches='tight',facecolor='white')
    plt.savefig("canada_3wkcasespop.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,axes=plt.subplots(figsize=(14,10))
    for province in ca_deaths:
        place=province
        if province != "Total" and ca_deaths[province][-1]>25:
            y = ca_deaths[province]/provincepops[province]*1e6
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y)-7)-(len(y)-6),day5avg(np.diff(y)),marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,day5avg(np.diff(y))[-1]))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(0.0,50)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("New Deaths per 1M")
    plt.title("Canada Daily Deaths"+latestglobal)
    plt.savefig("canada_dailydeathspop.png",bbox_inches='tight',facecolor='white')
    plt.savefig("canada_dailydeathspop.pdf",bbox_inches='tight')
    plt.close('all')
    
    
    fig,axes=plt.subplots(figsize=(14,10))
    for province in ca_deaths:
        place=province
        if province != "Total" and ca_deaths[province][-1]>25:
            y = ca_deaths[province]/provincepops[province]*1e6
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y)-7)-(len(y)-6),day5avg(np.diff(y)),marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,day5avg(np.diff(y))[-1]))
    #plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(0.0,50)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("New Deaths per 1M")
    plt.title("Canada Daily Deaths"+latestglobal)
    plt.savefig("canada_dailydeathspop_log.png",bbox_inches='tight',facecolor='white')
    plt.savefig("canada_dailydeathspop_log.pdf",bbox_inches='tight')
    plt.close('all')
    
    n=0
    labels=[]
    ptotals = {}
    for k in ca_deaths:
        province=k
        if province != "Total" and ca_deaths[province][-1]>2:
            ptotals[k] = ca_deaths[k][-1]/provincepops[k]*1e3
    fig,ax=plt.subplots(figsize=(12,4))
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        plt.bar(n,ptotals[k])
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.set_ylim(0,100)
    plt.ylabel("Deaths per 1000")
    #plt.yscale('log')
    plt.title("COVID-19 Death Toll"+latestglobal)
    maxd = round(ptotals[sorted(ptotals,key=ptotals.get,reverse=True)[0]])
    
    n=0
    #labels=[]
    ptotals = {}
    for k in ca_deaths:
        if k != "Total" and ca_deaths[k][-1]>25:
            ptotals[k] = provincepops[k]/ca_deaths[k][-1]
    
    worst = round(ptotals[sorted(ptotals,key=ptotals.get,reverse=True)[-1]])
    plt.annotate("Worst hit: 1 in %d dead in %s"%(worst,labels[0]),(1,maxd*1.1))
    plt.savefig("canada_deathtoll.png",bbox_inches='tight',facecolor='white')
    plt.savefig("canada_deathtoll.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(14,12))
    for place in ca_deaths:
        province=place
        if province != "Total" and ca_deaths[province][-1]>3: #Only plot places with >20 deaths
            y = day5avg(np.diff(ca_deaths[place]))
            #y = y[y>=800]
            #yp = y[0]
            #m = y[1]-y[0]
            #x0 = (3.0-yp)/m
            #x = np.arange(len(y)-1)-x0
            active = active3wk(canada[place])
            lst = ":"
            nactive = len(active)
            diffn = len(y)-nactive
            y = y[diffn:]/(active/21.0)
            alpha=0.5
            if place in ca_deaths:#["Minnesota","Wisconsin","New York","New Jersey","Oregon","Florida","Michigan","Connecticut","Massachusetts"]:
                lst='-'
                alpha=0.1
                plt.annotate(place,(len(y),y[-1]),xytext=(len(y),y[-1]),clip_on=True)
                    
            plt.plot(np.array(range(len(y))),y,marker='.',label=place,linestyle=lst,alpha=alpha,color='k')
            
    #plt.xscale('log')
    plt.yscale('log')
    #plt.legend(loc='best')
    plt.xlabel("Time [days]")
    plt.ylabel("Mortality Rate")
    plt.title("US States Mortality Rate"+latestusa)
    plt.savefig("ca_dailymortality.png",bbox_inches='tight',facecolor='white')
    plt.savefig("ca_dailymortality.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(14,12))
    for place in canada:
        province=place
        if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total": #Only plot places with >20 deaths
            y = canada[place]#/float(provincepops[place])*1e5
            #y = y[y>=800]
            #yp = y[0]
            #m = y[1]-y[0]
            #x0 = (3.0-yp)/m
            #x = np.arange(len(y)-1)-x0
                    
            r,p,l = Rt(day5avg(np.diff(y)),interval=7)
            r = week2avg(r)[:-13]
            z=active3wk(y)[-len(r):]
            z*=r
            x=np.array(range(len(r)))-len(r)
            lst = ":"
            alpha=0.5
            if place in canada:#["Minnesota","Wisconsin","New York","New Jersey","Oregon","Florida","Michigan","Connecticut","Massachusetts"]:
                lst='-'
                alpha=1.0
                plt.annotate(place,(0.25,max(r[-1],1)),xytext=(0.25,max(1,r[-1])),clip_on=True)
            plt.plot(x,r,marker='.',label=place,linestyle=lst,alpha=alpha)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(0.0,50)
    plt.axhline(1.0,linestyle=':',color='k')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("Days before Present")
    plt.ylabel("Effective Reproductive Number $R_t$ (14-day Average)")
    plt.title("Canada Transmission"+latestglobal)
    plt.savefig("canada_rt.png",bbox_inches='tight',facecolor='white')
    plt.savefig("canada_rt.pdf",bbox_inches='tight')
    plt.close('all')
    
    _log("/home/adivp416/public_html/covid19/reportlog.txt","US states plotted. \t%s"%systime.asctime(systime.localtime()))
    n=0
    labels=[]
    ptotals = {}
    active = {}
    nmax = 0
    for place in canada:
        province=place
        if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total": #Only plot places with >20 deaths
            y = canada[place]
            active[place] = active3wk(y)[-1]/float(provincepops[place])
            nmax = max(nmax,active[place])
            y = day5avg(np.diff(y[y>10]))
            r,p,l = Rt(y,interval=7)
            r = week2avg(r)
            ptotals[place] = r[-1]
    
    fig,ax=plt.subplots(figsize=(14,4))
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        if ptotals[k]>1.0:
            color='orange'
        elif ptotals[k]==1.0:
            color='blue'
        else:
            color='green'
        plt.bar(n,ptotals[k],color=color,edgecolor='k',alpha=max(0.05,active[k]/float(nmax)))
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.set_ylim(0,100)
    plt.ylabel("2-week Average R$_t$")
    #plt.yscale('log')
    #plt.ylim(5.0e-3,50.0)
    plt.axhline(1.0,linestyle='--',color='k')
    plt.title("Canada COVID-19 Reproductive Numbers, Opacity Weighted by Virus Prevalence"+latestglobal)
    plt.xlim(-1.0,n+1.0)
    plt.savefig("caprov_rt_snapshot.png",bbox_inches='tight',facecolor='white')
    plt.savefig("caprov_rt_snapshot.pdf",bbox_inches='tight')
    plt.close('all')
    
    n=0
    labels=[]
    ptotals = {}
    active = {}
    nmax = 0
    for place in canada:
        province=place
        if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total": #Only plot places with >20 deaths
            y = canada[place]
            active[place] = active3wk(y)[-1]/float(provincepops[place])
            nmax = max(nmax,active[place])
            y = day5avg(np.diff(y[y>10]))
            r,p,l = Rt(y,interval=7)
            r = week2avg(r)
            ptotals[place] = week2avg(np.gradient(r))[-1]
    
    fig,ax=plt.subplots(figsize=(14,4))
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        if ptotals[k]>0.0:
            color='orange'
        elif ptotals[k]==0.0:
            color='blue'
        else:
            color='green'
        plt.bar(n,ptotals[k],color=color,edgecolor='k',alpha=max(0.05,active[k]/float(nmax)))
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.set_ylim(0,100)
    plt.ylabel("2-week Average Derivative of <R$_t$>\n[Additional average infections per case per day]")
    #plt.yscale('log')
    #plt.ylim(5.0e-3,50.0)
    plt.axhline(0.0,linestyle='--',color='k')
    plt.title("Canada COVID-19 Reproductive Number Derivative (Transmission Trend), Opacity Weighted by Virus Prevalence"+latestglobal)
    plt.xlim(-1.0,n+1.0)
    plt.savefig("caprov_drt_snapshot.png",bbox_inches='tight',facecolor='white')
    plt.savefig("caprov_drt_snapshot.pdf",bbox_inches='tight')
    plt.close('all')
    
    
    #Begin doing US
    
    
    fig,ax=plt.subplots(figsize=(14,14))
    for place in usa:
        if usa[place][-1]>=20 and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place: #Only plot places with >20 confirmed cases
            plt.plot(usa[place]/float(statepops[place])*1e2,marker='.',label=place)
            coords = (len(usa[place]),usa[place][-1]/float(statepops[place])*1e2)
            plt.annotate(place,coords,xytext=coords)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlim(30,len(usa["Total"]))
    plt.ylim(0.1,30.0)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("Time [days]")
    plt.ylabel("Confirmed Cases (percentage of population)")
    plt.title("United States Confirmed Cases"+latestusa)
    plt.savefig("usconfirmed.png",bbox_inches='tight',facecolor='white')
    plt.savefig("usconfirmed.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,axes=plt.subplots(figsize=(14,10))
    for province in usa:
        place=province
        if us_deaths[place][-1]>=20 and place!="Total" and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place and province != "Total" and usa[province][-1]>150:
            y = usa[province]/statepops[province]*1e5
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y)-7)-(len(y)-6),day5avg(np.diff(y)),marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,day5avg(np.diff(y))[-1]))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(1.0e-2,100)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("New Cases per 100k")
    plt.title("USA Daily Cases"+latestusa)
    plt.savefig("usa_dailycasespop.png",bbox_inches='tight',facecolor='white')
    plt.savefig("usa_dailycasespop.pdf",bbox_inches='tight')
    plt.close('all')
    
    
    fig,axes=plt.subplots(figsize=(14,10))
    for province in usa:
        place=province
        if us_deaths[place][-1]>=20 and place!="Total" and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place and province != "Total" and usa[province][-1]>150:
            y = usa[province]/statepops[province]*1e5
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y)-7)-(len(y)-6),day5avg(np.diff(y)),marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,day5avg(np.diff(y))[-1]))
    #plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(1.0e-2,100)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("New Cases per 100k")
    plt.title("USA Daily Cases"+latestusa)
    plt.savefig("usa_dailycasespop_log.png",bbox_inches='tight',facecolor='white')
    plt.savefig("usa_dailycasespop_log.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,axes=plt.subplots(figsize=(14,10))
    for province in usa:
        place=province
        if us_deaths[place][-1]>=20 and place!="Total" and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place and province != "Total" and usa[province][-1]>150:
            y = active3wk(usa[province])/statepops[province]*1e2
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y))-len(y),y,marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,y[-1]))
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(1.0,2.0e3)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("3-wk Cumulative Cases [Percent of Population]")
    plt.title("USA Active Cases"+latestusa)
    plt.savefig("usa_3wkcasespop.png",bbox_inches='tight',facecolor='white')
    plt.savefig("usa_3wkcasespop.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,axes=plt.subplots(figsize=(14,10))
    for province in usa:
        place=province
        if us_deaths[place][-1]>=20 and place!="Total" and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place and province != "Total" and usa[province][-1]>150:
            y = us_deaths[province]/statepops[province]*1e6
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y)-7)-(len(y)-6),day5avg(np.diff(y)),marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,day5avg(np.diff(y))[-1]))
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylim(0.0,50)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("New Deaths per 1M")
    plt.title("USA Daily Deaths"+latestusa)
    plt.savefig("usa_dailydeathspop.png",bbox_inches='tight',facecolor='white')
    plt.savefig("usa_dailydeathspop.pdf",bbox_inches='tight')
    plt.close('all')
    
    
    fig,axes=plt.subplots(figsize=(14,10))
    for province in usa:
        place=province
        if us_deaths[place][-1]>=20 and place!="Total" and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place and province != "Total" and usa[province][-1]>150:
            y = us_deaths[province]/statepops[province]*1e6
            #y = y[y>=150]
            yp = y[0]
            m = y[1]-y[0]
            x0 = (150.0-yp)/m
            x = np.arange(len(y)-1)-x0
            plt.plot(np.arange(len(y)-7)-(len(y)-6),day5avg(np.diff(y)),marker='.',label=province,linestyle=':')
            plt.annotate(province,(0,day5avg(np.diff(y))[-1]))
    #plt.xscale('log')
    plt.yscale('log')
    #plt.ylim(0.0,50)
    #plt.legend(loc='best')
    plt.xlabel("Time before Present [days]")
    plt.ylabel("New Deaths per 1M")
    plt.title("USA Daily Deaths"+latestusa)
    plt.savefig("usa_dailydeathspop_log.png",bbox_inches='tight',facecolor='white')
    plt.savefig("usa_dailydeathspop_log.pdf",bbox_inches='tight')
    plt.close('all')
    
    n=0
    labels=[]
    ptotals = {}
    for k in usa:
        if "Princess" not in k and "Virgin Islands" not in k and "Recovered" not in k and "Virgin Islands" not in k and "Guam" not in k\
        and "Military" not in k and "Prisons" not in k and "Hospitals" not in k:
            ptotals[k] = us_deaths[k][-1]/statepops[k]*1e3
    fig,ax=plt.subplots(figsize=(12,4))
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        plt.bar(n,ptotals[k])
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.set_ylim(0,100)
    plt.ylabel("Deaths per 1000")
    #plt.yscale('log')
    plt.title("COVID-19 Death Toll"+latestusa)
    maxd = round(ptotals[sorted(ptotals,key=ptotals.get,reverse=True)[0]])
    
    n=0
    #labels=[]
    ptotals = {}
    for k in usa:
        if "Princess" not in k and "Virgin Islands" not in k and "Recovered" not in k and "Virgin Islands" not in k and "Guam" not in k\
        and "Military" not in k and "Prisons" not in k and "Hospitals" not in k:
            ptotals[k] = statepops[k]/us_deaths[k][-1]
    
    worst = round(ptotals[sorted(ptotals,key=ptotals.get,reverse=True)[-1]])
    plt.annotate("Worst hit: 1 in %d dead in %s"%(worst,labels[0]),(1,maxd*1.1))
    plt.savefig("usa_deathtoll.png",bbox_inches='tight',facecolor='white')
    plt.savefig("usa_deathtoll.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(14,12))
    for place in us_deaths:
        if us_deaths[place][-1]>=500 and place!="Total" and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place: #Only plot places with >20 deaths
            y = day5avg(np.diff(us_deaths[place]))
            #y = y[y>=800]
            #yp = y[0]
            #m = y[1]-y[0]
            #x0 = (3.0-yp)/m
            #x = np.arange(len(y)-1)-x0
            active = active3wk(usa[place])
            lst = ":"
            nactive = len(active)
            diffn = len(y)-nactive
            y = y[diffn:]/(active/21.0)
            alpha=0.5
            if place in us_deaths:#["Minnesota","Wisconsin","New York","New Jersey","Oregon","Florida","Michigan","Connecticut","Massachusetts"]:
                lst='-'
                alpha=0.1
                plt.annotate(place,(len(y),y[-1]),xytext=(len(y),y[-1]),clip_on=True)
                    
            plt.plot(np.array(range(len(y))),y,marker='.',label=place,linestyle=lst,alpha=alpha,color='k')
            
    #plt.xscale('log')
    plt.yscale('log')
    #plt.legend(loc='best')
    plt.xlabel("Time [days]")
    plt.ylabel("Mortality Rate")
    plt.title("US States Mortality Rate"+latestusa)
    plt.savefig("us_dailymortality.png",bbox_inches='tight',facecolor='white')
    plt.savefig("us_dailymortality.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(14,12))
    for place in usa:
        if usa[place][-1]>=500 and place!="Total" and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place: #Only plot places with >20 deaths
            y = usa[place]#/float(statepops[place])*1e5
            #y = y[y>=800]
            #yp = y[0]
            #m = y[1]-y[0]
            #x0 = (3.0-yp)/m
            #x = np.arange(len(y)-1)-x0
                    
            r,p,l = Rt(day5avg(np.diff(y)),interval=7)
            r = week2avg(r)[:-13]
            z=active3wk(y)[-len(r):]
            z*=r
            x=np.array(range(len(r)))-len(r)
            lst = ":"
            alpha=0.5
            if place in usa:#["Minnesota","Wisconsin","New York","New Jersey","Oregon","Florida","Michigan","Connecticut","Massachusetts"]:
                lst='-'
                alpha=1.0
                plt.annotate(place,(0.25,max(r[-1],1)),xytext=(0.25,max(1,r[-1])),clip_on=True)
            plt.plot(x,r,marker='.',label=place,linestyle=lst,alpha=alpha)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.ylim(0.0,50)
    plt.axhline(1.0,linestyle=':',color='k')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xlabel("Days before Present")
    plt.ylabel("Effective Reproductive Number $R_t$ (14-day Average)")
    plt.title("US Transmission"+latestusa)
    plt.savefig("usa_rt.png",bbox_inches='tight',facecolor='white')
    plt.savefig("usa_rt.pdf",bbox_inches='tight')
    plt.close('all')
    
    
    _log("/home/adivp416/public_html/covid19/reportlog.txt","US nation-wide data plotted. \t%s"%systime.asctime(systime.localtime()))
    
    for place in usa:
        if usa[place][-1]>=500 and place!="Total" and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place: #Only plot places with >20 deaths
            plot_stateRt(place,usa,latestusa)
    
    _log("/home/adivp416/public_html/covid19/reportlog.txt","US states plotted. \t%s"%systime.asctime(systime.localtime()))
    n=0
    labels=[]
    ptotals = {}
    active = {}
    nmax = 0
    for place in usa:
        if usa[place][-1]>=500 and place!="Total" and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place: #Only plot places with >20 deaths
            y = usa[place]
            active[place] = active3wk(y)[-1]/float(statepops[place])
            nmax = max(nmax,active[place])
            y = day5avg(np.diff(y[y>10]))
            r,p,l = Rt(y,interval=7)
            r = week2avg(r)
            ptotals[place] = r[-1]
    
    fig,ax=plt.subplots(figsize=(14,4))
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        if ptotals[k]>1.0:
            color='orange'
        elif ptotals[k]==1.0:
            color='blue'
        else:
            color='green'
        plt.bar(n,ptotals[k],color=color,edgecolor='k',alpha=max(0.05,active[k]/float(nmax)))
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.set_ylim(0,100)
    plt.ylabel("2-week Average R$_t$")
    #plt.yscale('log')
    #plt.ylim(5.0e-3,50.0)
    plt.axhline(1.0,linestyle='--',color='k')
    plt.title("US COVID-19 Reproductive Numbers, Opacity Weighted by Virus Prevalence"+latestusa)
    plt.xlim(-1.0,n+1.0)
    plt.savefig("usstates_rt_snapshot.png",bbox_inches='tight',facecolor='white')
    plt.savefig("usstates_rt_snapshot.pdf",bbox_inches='tight')
    plt.close('all')
    
    n=0
    labels=[]
    ptotals = {}
    active = {}
    nmax = 0
    for place in usa:
        if usa[place][-1]>=500 and place!="Total" and "Princess" not in place and "Virgin Islands" not in place and "Military" not in place\
        and "Recovered" not in place and "Prisons" not in place and "Hospitals" not in place: #Only plot places with >20 deaths
            y = usa[place]
            active[place] = active3wk(y)[-1]/float(statepops[place])
            nmax = max(nmax,active[place])
            y = day5avg(np.diff(y[y>10]))
            r,p,l = Rt(y,interval=7)
            r = week2avg(r)
            ptotals[place] = week2avg(np.gradient(r))[-1]
    
    fig,ax=plt.subplots(figsize=(14,4))
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        if ptotals[k]>0.0:
            color='orange'
        elif ptotals[k]==0.0:
            color='blue'
        else:
            color='green'
        plt.bar(n,ptotals[k],color=color,edgecolor='k',alpha=max(0.05,active[k]/float(nmax)))
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.set_ylim(0,100)
    plt.ylabel("2-week Average Derivative of <R$_t$>\n[Additional average infections per case per day]")
    #plt.yscale('log')
    #plt.ylim(5.0e-3,50.0)
    plt.axhline(0.0,linestyle='--',color='k')
    plt.title("US COVID-19 Reproductive Number Derivative (Transmission Trend), Opacity Weighted by Virus Prevalence"+latestusa)
    plt.xlim(-1.0,n+1.0)
    plt.savefig("usstates_drt_snapshot.png",bbox_inches='tight',facecolor='white')
    plt.savefig("usstates_drt_snapshot.pdf",bbox_inches='tight')
    plt.close('all')

    for state in uskeys:
        plotStateOrProvince(state,"United States",usa,us_deaths,
                                extract_country(dataset,"US")["Total"]/float(countrypops["US"]),
                                extract_country(ddataset,"US")["Total"]/float(countrypops["US"]),
                                statepops[state],latestusa)

    _log("/home/adivp416/public_html/covid19/reportlog.txt","Finished state-level data. Starting counties. \t%s"%systime.asctime(systime.localtime()))
    
    _log("/home/adivp416/public_html/covid19/reportlog.txt","Moving on to global data. \t%s"%systime.asctime(systime.localtime()))

    fig,ax=plt.subplots(figsize=(12,12))
    tmax = 0
    tmin = 0
    nmax = 0
    for country in countries:
        try:
            cdata = extract_country(ddataset,country)
            if cdata["Total"][-1]>=100:
                y = cdata["Total"]
                y = y/countrypops[country]*1.0e3
                x = np.arange(len(y))-len(y)
                tmax = max(tmax,len(y)*1.1)
                tmin = min(tmin,x.min())
                nmax = max(nmax,1.1*max(y))
                plt.plot(x,y,label=country,marker='.')
                coords = (x[-1]+0.5,y[-1])
                plt.annotate(country,coords,xytext=coords,clip_on=True)
        except:
            traceback.print_exc()
    cdata = extract_country(ddataset,"US")
    y = cdata["Total"]
    y = y/countrypops["United States"]*1.0e3
    x = np.arange(len(y))-len(y)
    tmax = max(tmax,len(y)*1.1)
    tmin = min(tmin,x.min())
    nmax = max(nmax,1.1*max(y))
    plt.plot(x,y,label=country,marker='.')
    coords = (x[-1]+0.5,y[-1])
    plt.annotate("USA",coords,xytext=coords,clip_on=True)
    plt.ylim(1.0e-3,nmax)
    plt.xlim(tmin,250)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylabel("Deaths Per 1000 People")
    plt.xlabel("Days before Present")
    plt.title("COVID-19 Deaths per Thousand"+latestglobal)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("deathnumberspop.png",bbox_inches='tight',facecolor='white')
    plt.savefig("deathnumberspop.pdf",bbox_inches='tight')
    plt.close('all')
    
    fig,ax=plt.subplots(figsize=(12,12))
    tmax = 0
    tmin = 0
    nmax = 0
    for country in countries:
        try:
            cdata = extract_country(ddataset,country)
            if cdata["Total"][-1]>=100:
                y = cdata["Total"]
                y = y/countrypops[country]*1.0e3
                x = np.arange(len(y))-len(y)
                tmax = max(tmax,len(y)*1.1)
                tmin = min(tmin,x.min())
                nmax = max(nmax,1.1*max(y))
                plt.plot(x,y,label=country,marker='.')
                coords = (x[-1]+0.5,y[-1])
                plt.annotate(country,coords,xytext=coords,clip_on=True)
        except:
            traceback.print_exc()
    cdata = extract_country(ddataset,"US")
    y = cdata["Total"]
    y = y/countrypops["United States"]*1.0e3
    x = np.arange(len(y))-len(y)
    tmax = max(tmax,len(y)*1.1)
    tmin = min(tmin,x.min())
    nmax = max(nmax,1.1*max(y))
    plt.plot(x,y,label=country,marker='.')
    coords = (x[-1]+0.5,y[-1])
    plt.annotate("USA",coords,xytext=coords,clip_on=True)
    plt.ylim(1.0e-3,nmax)
    plt.xlim(tmin,250)
    #plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("Deaths Per 1000 People")
    plt.xlabel("Days before Present")
    plt.title("COVID-19 Deaths per Thousand"+latestglobal)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("deathnumberspop_log.png",bbox_inches='tight',facecolor='white')
    plt.savefig("deathnumberspop_log.pdf",bbox_inches='tight')
    plt.close('all')

    ptotals = {}
    for country in countries:
        try:
            cdata = extract_country(ddataset,country)
            if cdata["Total"][-1]>=25:
                y = cdata["Total"]
                ptotals[country] = y[-1]/float(countrypops[country])*1.0e3
        except Exception as e:
            traceback.print_exc()
    cdata = extract_country(ddataset,"US")
    y = cdata["Total"]
    y = y/float(countrypops["United States"])*1.0e3
    ptotals["United States"] = y[-1]
    print("US deaths per 1000:",y[-1])
    n=0
    labels = []
    fig,ax=plt.subplots(figsize=(24,4))
    for k in (sorted(ptotals, key=ptotals.get,reverse=True))[::-1]:
        labels.append(k)
        plt.bar(n,1.0/(ptotals[k]*1.0e-3))
        if k=="United States" or k=="Canada":
            plt.scatter(n,1.0/(ptotals[k]*1.0e-3)*1.6,color='k',marker='*')
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.set_ylim(0,100)
    plt.ylabel("1 death per [x]")
    plt.yscale('log')
    plt.title("Global COVID-19 Death Toll"+latestglobal)
    plt.xlim(-1.0,n+1.0)
    print(labels[-1],ptotals[labels[-1]]*1.0e-3)
    print("US",ptotals["United States"]*1.0e-3)
    print("CA",ptotals["Canada"]*1.0e-3)
    print("AU",ptotals["Australia"]*1.0e-3)
    print("NZ",ptotals["New Zealand"]*1.0e-3)
    plt.annotate("Worst-hit is %s with 1 death per %d people."%(labels[-1],int(round(1.0/(ptotals[labels[-1]]*1.0e-3))))+"\n"+
                 "In the US, 1 in %d people have died."%(int(round(1.0/(ptotals["United States"]*1.0e-3))))+"\n"+
                 "In Canada, 1 in %d people have died."%(int(round(1.0/(ptotals["Canada"]*1.0e-3))))+"\n"+
                 "In Australia, 1 in %d people have died."%(int(round(1.0/(ptotals["Australia"]*1.0e-3))))+"\n"+
                 "In New Zealand, 1 in %d people have died."%(int(round(1.0/(ptotals["New Zealand"]*1.0e-3)))),
                 (30,int(round(1.0/(ptotals[labels[0]]*1.0e-3*0.1)))),clip_on=True)
    plt.savefig("globaldeathtoll.png",bbox_inches='tight',facecolor='white')
    plt.savefig("globaldeathtoll.pdf",bbox_inches='tight')
    plt.close('all')

    interestcountries = ["Sweden","Finland","United Kingdom","France","Spain","Portugal","Belgium",
                         "Norway","Iceland","Germany","Italy","Greece","New Zealand","China","Denmark",
                         "Switzerland","Israel"]

    fig,ax=plt.subplots(figsize=(12,12))
    tmax = 0
    nmax = 0
    for country in countries:
        try:
            cdata = extract_country(ddataset,country)
            if cdata["Total"][-1]>=10 and "Princess" not in country and "Olympics" not in country:
                y = cdata["Total"]
                y = day5avg(np.diff(y/countrypops[country]*1.0e6))
                #y = y[y>=1]
                #yp = y[0]
                #m,b = np.polyfit(range(5),np.log10(y[:5]),1)
                #x0 = -np.log10(yp)/m
                x = np.arange(len(y))#-x0
                #tmax = max(tmax,len(y)*1.1)
                nmax = max(nmax,1.1*max(y))
                alpha=0.1
                color='gray'
                label=None
                if country in interestcountries:
                    alpha=1.0
                    color=None
                    label=country
                plt.plot(x,y,marker='.',alpha=alpha,color=color,label=label)
                coords = (x[-1]+0.5,y[-1])
                plt.annotate(country,coords,xytext=coords,alpha=alpha,clip_on=True)
        except Exception as e:
            traceback.print_exc()
    cdata = extract_country(ddataset,"US")
    y = cdata["Total"]
    y = day5avg(np.diff(y/countrypops["United States"]*1.0e6))
    #y = y[y>=1]
    #yp = y[0]
    #m,b = np.polyfit(range(5),np.log10(y[:5]),1)
    #x0 = -np.log10(yp)/m
    x = np.arange(len(y))#-x0
    tmax = max(tmax,len(y)*1.1)
    nmax = max(nmax,1.1*max(y))
    plt.plot(x,y,marker='.',label="USA")
    coords = (x[-1]+0.5,y[-1]*0.9)
    plt.annotate("USA",coords,xytext=coords,clip_on=True)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


    plt.ylim(1.0e-2,nmax)
    plt.xlim(1,tmax*1.1)
    #plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("Deaths Per 1 Million People")
    plt.xlabel("Time [days]")
    plt.title("COVID-19 Deaths per Million per Day"+latestglobal)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("selectcountriesdailydeaths.png",bbox_inches='tight',facecolor='white')
    plt.savefig("selectcountriesdailydeaths.pdf",bbox_inches='tight')
    plt.close('all')

    n=0
    labels=[]
    ptotals = {}
    for country in countries:
        try:
            cdata = extract_country(ddataset,country)
            if cdata["Total"][-1]>=25:
                y = cdata["Total"]
                y = day5avg(np.diff(y/countrypops[country]*1.0e6))
                ptotals[country] = y[-1]
        except Exception as e:
            traceback.print_exc()
    cdata = extract_country(ddataset,"US")
    y = cdata["Total"]
    y = day5avg(np.diff(y/countrypops["United States"]*1.0e6))
    ptotals["United States"] = y[-1]
    fig,ax=plt.subplots(figsize=(24,4))
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        plt.bar(n,ptotals[k])
        if k=="United States" or k=="Canada":
            plt.scatter(n,ptotals[k]*1.5,color='k',marker='*')
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.set_ylim(0,100)
    plt.ylabel("Deaths per Million per Day")
    plt.yscale('log')
    plt.ylim(5.0e-3,50.0)
    plt.title("Global COVID-19 Daily Deaths"+latestglobal)
    plt.xlim(-1.0,n+1.0)
    plt.savefig("globaldailydeaths_recent.png",bbox_inches='tight',facecolor='white')
    plt.savefig("globaldailydeaths_recent.pdf",bbox_inches='tight')
    plt.close('all')

    fig,ax=plt.subplots(figsize=(12,12))
    maxy=0
    xmin = 0
    for country in countries:
        try:
            cdata = extract_country(dataset,country)
            if cdata["Total"][-1]>=100:
                y = cdata["Total"]
                #y = y[y>=150]#/countrypops[country]*1.0e6
                #yp = y[0]
                #m = y[1]-y[0]
                #x0 = (150.0-yp)/m
                ya = day5avg(np.diff(y))
                x = np.arange(len(ya))-len(ya)#-x0
                xmin = min(xmin,x.min())
                plt.plot(x,ya,label=country,marker='.')
                coords = (1,ya[-1])
                maxy = max(maxy,np.nanmax(np.diff(y)))
                #coords = (len(x),y[-1])
                if np.diff(y)[-1]>10:
                    #print(country,coords)
                    plt.annotate(country,coords,xytext=coords,clip_on=True)
        except Exception as e:
            traceback.print_exc()
    #plt.annotate("USA",coords,xytext=coords)
    #plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1.0,maxy*1.3)
    plt.xlim(xmin,-xmin*0.2)
    plt.ylabel("New Cases Per Day")
    plt.xlabel("Days Before Present")
    plt.title("COVID-19 Transmission"+latestglobal)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig("casetransmission.png",bbox_inches='tight',facecolor='white')
    plt.savefig("casetransmission.pdf",bbox_inches='tight')
    plt.close('all')

    n=0
    labels=[]
    ptotals = {}
    for country in countries:
        try:
            cdata = extract_country(dataset,country)
            if cdata["Total"][-1]>=25 and "Princess" not in country and "Olympics" not in country:
                y = cdata["Total"]
                y = day5avg(np.diff(y/float(countrypops[country])*1.0e6))
                ptotals[country] = y[-1]
                if country=="Germany":
                    print("Germany:",y[-1],y[-5:])
                    print("Germany Daily:",np.diff(cdata["Total"])[-5:])
        except Exception as e:
            traceback.print_exc()
    cdata = extract_country(dataset,"US")
    y = cdata["Total"]
    y = day5avg(np.diff(y/countrypops["United States"]*1.0e6))
    ptotals["United States"] = y[-1]
    fig,ax=plt.subplots(figsize=(24,4))
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        plt.bar(n,ptotals[k])
        if k=="United States" or k=="Canada":
            plt.scatter(n,ptotals[k]*1.5,color='k',marker='*')
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.set_ylim(0,100)
    plt.ylabel("Cases per Million per Day")
    plt.yscale('log')
    #plt.ylim(5.0e-3,50.0)
    plt.title("Global COVID-19 Daily Cases"+latestglobal)
    plt.xlim(-1.0,n+1.0)
    plt.savefig("globalcases_pop_recent.png",bbox_inches='tight',facecolor='white')
    plt.savefig("globalcases_pop_recent.pdf",bbox_inches='tight')
    plt.close('all')

    n=0
    labels=[]
    ptotals = {}
    active = {}
    nmax = 0
    for country in countries:
        try:
            if country!="US":
                cdata = extract_country(dataset,country)
                if cdata["Total"][-1]>=25 and "Princess" not in country and "Olympics" not in country:
                    y = cdata["Total"]
                    active[country] = active3wk(y)[-1]/countrypops[country]
                    nmax = max(nmax,active[country])
                    r,p,l = Rt(day5avg(np.diff(y)))
                    r = week2avg(r)
                    ptotals[country] = r[-1]
        except Exception as e:
            traceback.print_exc()
    cdata = extract_country(dataset,"US")
    y = cdata["Total"]
    active["United States"] = active3wk(y)[-1]/countrypops["United States"]
    nmax = max(nmax,active["United States"])
    print(nmax)
    r,p,l = Rt(day5avg(np.diff(y)))
    r = week2avg(r)
    ptotals["United States"] = r[-1]
    fig,ax=plt.subplots(figsize=(24,4))
    for k in sorted(ptotals, key=ptotals.get,reverse=True):
        labels.append(k)
        if ptotals[k]>1.0:
            color='orange'
        elif ptotals[k]==1.0:
            color='blue'
        else:
            color='green'
        plt.bar(n,ptotals[k],color=color,edgecolor='k',alpha=max(0.05,active[k]/float(nmax)))
        if k=="United States" or k=="Canada":
            plt.scatter(n,ptotals[k]+0.2,color='k',marker='*')
        n+=1
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels,rotation='vertical')
    #ax.set_ylim(0,100)
    plt.ylabel("2-week Average R$_t$")
    #plt.yscale('log')
    #plt.ylim(5.0e-3,50.0)
    plt.axhline(1.0,linestyle='--',color='k')
    plt.title("Global COVID-19 Reproductive Numbers, Opacity Weighted by Virus Prevalence"+latestglobal)
    plt.xlim(-1.0,n+1.0)
    plt.savefig("global_rt.png",bbox_inches='tight',facecolor='white')
    plt.savefig("global_rt.pdf",bbox_inches='tight')
    plt.close('all')

    _log("/home/adivp416/public_html/covid19/reportlog.txt","Global data plotted. \t%s"%systime.asctime(systime.localtime()))

    for country in countries:
        try:
            if cdata["Total"][-1]>=25 and "Princess" not in country and "Olympics" not in country and "Zaandam" not in country:
                country_summary(country,dataset,ddataset,countrypops,latestglobal)
                _log("/home/adivp416/public_html/covid19/reportlog.txt","%s data plotted. \t%s"%(country,systime.asctime(systime.localtime())))
                makehtml.makeCountry(country)
            
        except Exception as e:
            traceback.print_exc()
            _log("/home/adivp416/public_html/covid19/reportlog.txt","No data or broken data for %s \t%s"%(country,systime.asctime(systime.localtime())))
    
    
    for country in sorted(countries):
        if os.path.exists("%s_summary.png"%country):
            makehtml.makeCountry(country)
    
    with open("index.html","r") as indexf:
        index = indexf.read().split('\n')
    html = []
    skipnext=False
    for line in index:
        if not skipnext:
            if "<!-- GLOBALFORM -->" in line:
                html.append(line)
                skipnext=True
                for country in sorted(countries):
                    if os.path.exists("%s_country.html"%(country.replace('"','').replace('&','and').replace(",","").replace("/","_").replace(" ","_").replace("-","_"))):
                        html.append('<!--GL-->\t\t\t<option value="%s">%s</option>'%(country,country))
            elif "<!-- PLACEHOLDER -->" in line:
                skipnext=True
            elif "<option" in line and "<!--GL-->" in line:
                pass #skip thi line too 
            else:
                html.append(line)
        else:
            if "<!-- TRIPWIRE -->" in line:
                html.append(line)
            skipnext=False
    with open("index.html","w") as indexf:
        indexf.write("\n".join(html))
    
    _log("/home/adivp416/public_html/covid19/reportlog.txt","Individual countries plotted. \t%s"%systime.asctime(systime.localtime()))
    
def processcounties(state):
        
    import makehtml
        
    state = state.replace("_"," ")
        
    statesetf = "statepopulations.csv"
    with open(statesetf,"r") as df:
        stateset = df.read().split('\n')[2:]
    if stateset[-1] == "":
        stateset = stateset[:-1]
    statepops = {}
    for line in stateset:
        linedata = line.split(',')
        name = linedata[2]
        popx = float(linedata[3])
        statepops[name] = popx
        
    _log("/home/adivp416/public_html/covid19/reportlog.txt","Static CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
        
    globalconf = "github/time_series_covid19_confirmed_global.csv"

    with open(globalconf,"r") as df:
        dataset = df.read().split('\n')
    header = dataset[0].split(',')
    dataset = dataset[1:]
    if dataset[-1]=='':
        dataset = dataset[:-1]
    latestglobal = header[-1]
    

    usacsv = []
    with open("github/time_series_covid19_confirmed_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usacsv.append(row)
            
    usa = extract_usa(usacsv)
            
    _log("/home/adivp416/public_html/covid19/reportlog.txt","Dynamic CSVs loaded. \t%s"%systime.asctime(systime.localtime()))

    latestusa = usacsv[0][-1]
    usatime = latestusa.split("/")
    latestusa = date(2000+int(usatime[2]),int(usatime[0]),int(usatime[1]))
    latestusa = "\nAs of "+' '.join([x for i,x in enumerate(latestusa.ctime().split()) if i!=3])

    counties = get_counties(usacsv,state=state)
    for county in counties:
        try:
            plotCounty(county,state,usacsv,usa,statepops[state],latestusa)
            _log("/home/adivp416/public_html/covid19/reportlog.txt","Finished %s County, %s. \t%s"%(county,state,systime.asctime(systime.localtime())))
            
            fstub1 = state.replace(" ","_").replace("&","and")
            fstub2 = str.title(county).replace('"','').replace('&','and').replace(",","").replace("/","_").replace(" ","_").replace("-","_")
            pathdir = "%s/%s"%(fstub1,fstub2)
            makehtml.makeCounty(county,state,pathdir)
        except:
            _log("/home/adivp416/public_html/covid19/reportlog.txt","Error encountered with %s County, %s:"%(county,state))
            print("Error encountered with %s County, %s:"%(county,state))
            traceback.print_exc()
            #raise

def makeshell():
    
    statesetf = "statepopulations.csv"
    with open(statesetf,"r") as df:
        stateset = df.read().split('\n')[2:]
    if stateset[-1] == "":
        stateset = stateset[:-1]
    statepops = {}
    for line in stateset:
        linedata = line.split(',')
        name = linedata[2]
        popx = float(linedata[3])
        statepops[name] = popx
        
    globalconf = "github/time_series_covid19_confirmed_global.csv"

    with open(globalconf,"r") as df:
        dataset = df.read().split('\n')
    header = dataset[0].split(',')
    dataset = dataset[1:]
    if dataset[-1]=='':
        dataset = dataset[:-1]
    latestglobal = header[-1]
    

    usacsv = []
    with open("github/time_series_covid19_confirmed_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usacsv.append(row)
         
    ddatasetf = "github/time_series_covid19_deaths_global.csv"

    with open(ddatasetf,"r") as df:
        ddataset = df.read().split('\n')
    header = ddataset[0].split(',')
    ddataset = ddataset[1:]
    if ddataset[-1]=='':
        ddataset = ddataset[:-1]
        
    usadcsv = []
    with open("github/time_series_covid19_deaths_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usadcsv.append(row)
    
    us_deaths = extract_usa(usadcsv)   
    usa = extract_usa(usacsv)
    
    uskeys = []
    for state in usa:
        if us_deaths[state][-1]>=20 and state!="Total" and "Princess" not in state\
            and "Virgin Islands" not in state and "Military" not in state\
            and "Recovered" not in state and "Prisons" not in state\
            and "Hospitals" not in state and state != "Total" and usa[state][-1]>150:
            uskeys.append(state)
            
    cdir = os.getcwd()
    
    for state in uskeys:
        place = state.replace(" ","_")
        text = ("#!/bin/bash \n"+
                "cd /home/adivp416/public_html/covid19/ \n"+
                "python %s/covid19report.py counties %s \n"%(cdir,place))
        with open("countyreport_%s.sh"%place,"w") as shellf:
            shellf.write(text)
        os.system("chmod a+x %s/countyreport_%s.sh"%(cdir,place))

def netcdf():
    import netCDF4 as nc
    
    ncd = nc.Dataset("adivparadise_covid19data.nc","w")
    
    countrysetf = "countrypopulations.csv"
    with open(countrysetf,"r") as df:
        countryset = df.read().split('\n')[1:]
    if countryset[-1] == "":
        countryset = countryset[:-1]
    countrypops = {}
    for line in countryset:
        linedata = line.split(',')
        name = linedata[0]
        popx = float(linedata[4])
        countrypops[name] = popx
    
    canadasetf = "provincepopulations.csv"
    with open(canadasetf,"r") as df:
        canadaset = df.read().split('\n')[2:]
    if canadaset[-1] == "":
        canadaset = canadaset[:-1]
    provincepops = {}
    for line in canadaset:
        linedata = line.split(',')
        name = linedata[1]
        popx = float(linedata[-3])
        provincepops[name] = popx
        
    statesetf = "statepopulations.csv"
    with open(statesetf,"r") as df:
        stateset = df.read().split('\n')[2:]
    if stateset[-1] == "":
        stateset = stateset[:-1]
    statepops = {}
    for line in stateset:
        linedata = line.split(',')
        name = linedata[2]
        popx = float(linedata[3])
        statepops[name] = popx
        
    #_log("/home/adivp416/public_html/covid19/reportlog.txt","Static CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
        
    globalconf = "github/time_series_covid19_confirmed_global.csv"

    with open(globalconf,"r") as df:
        dataset = df.read().split('\n')
    header = dataset[0].split(',')
    dataset = dataset[1:]
    if dataset[-1]=='':
        dataset = dataset[:-1]
    latestglobal = header[-1]
        
    countries = getcountries(dataset)
    

    usacsv = []
    with open("github/time_series_covid19_confirmed_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usacsv.append(row)
            
    usa = extract_usa(usacsv)

    canada = extract_country(dataset,"Canada")

    ddatasetf = "github/time_series_covid19_deaths_global.csv"

    with open(ddatasetf,"r") as df:
        ddataset = df.read().split('\n')
    header = ddataset[0].split(',')
    ddataset = ddataset[1:]
    if ddataset[-1]=='':
        ddataset = ddataset[:-1]
        
    usadcsv = []
    with open("github/time_series_covid19_deaths_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usadcsv.append(row)
            
    #_log("/home/adivp416/public_html/covid19/reportlog.txt","Dynamic CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
    
    ca_deaths = extract_country(ddataset,"Canada")
    us_deaths = extract_usa(usadcsv)

    timestamps = []
    rtimes = []
    ftimes = []
    htimes = []
    TOneighborhoods = {"units":{}}
    with open("toronto_cases.csv","r") as df:
        torontocsv = df.read().split('\n')[1:]
        while torontocsv[-1]=="":
            torontocsv = torontocsv[:-1]
        torontoraw = np.zeros(len(torontocsv))
        for line in range(len(torontoraw)):
            timestamp = torontocsv[line].split(',')[9].split('-')
            timestamps.append(date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])))
            entry = torontocsv[line].split(',')
            if entry[11]=="ACTIVE":
                pass
            elif entry[11]=="FATAL":
                ftimes.append(timestamps[-1])
            else:
                rtimes.append(timestamps[-1])
            if entry[15]=="Yes":
                htimes.append(timestamps[-1])
                
            if entry[4] not in TOneighborhoods["units"]:
                TOneighborhoods["units"][entry[4]] = {"CASES":[],"FATAL":[],"HOSPITALIZED":[],"RECOVERED":[]}
            TOneighborhoods["units"][entry[4]]["CASES"].append(timestamps[-1])
            if entry[11]=="ACTIVE":
                pass
            elif entry[11]=="FATAL":
                TOneighborhoods["units"][entry[4]]["FATAL"].append(timestamps[-1])
            else:
                TOneighborhoods["units"][entry[4]]["RECOVERED"].append(timestamps[-1])
            if entry[15]=="Yes":
                TOneighborhoods["units"][entry[4]]["HOSPITALIZED"].append(timestamps[-1])
            
    #for neighborhood in TOneighborhoods["units"]:
    #    TOneighborhoods["units"][neighborhood]["CASES"] = np.array(TOneighborhoods["units"][neighborhood]["CASES"])
    #    TOneighborhoods["units"][neighborhood]["FATAL"] = np.array(TOneighborhoods["units"][neighborhood]["FATAL"])
    #    TOneighborhoods["units"][neighborhood]["HOSPITALIZED"] = np.array(TOneighborhoods["units"][neighborhood]["HOSPITALIZED"])
    #    TOneighborhoods["units"][neighborhood]["RECOVERED"] = np.array(TOneighborhoods["units"][neighborhood]["RECOVERED"])
            
    #All cases
    origin = np.array(timestamps).min()
    for line in range(len(timestamps)):
        day = ((timestamps[line]-origin).days)
        torontoraw[line] = int(day)
    #Recovered cases
    torontorraw = np.zeros(len(rtimes))
    for line in range(len(rtimes)):
        day = ((rtimes[line]-origin).days)
        torontorraw[line] = int(day)
    #Hospitalized cases
    torontohraw = np.zeros(len(htimes))
    for line in range(len(htimes)):
        day = ((htimes[line]-origin).days)
        torontohraw[line] = int(day)
    #Fatal cases
    torontofraw = np.zeros(len(ftimes))
    for line in range(len(ftimes)):
        day = ((ftimes[line]-origin).days)
        torontofraw[line] = int(day)
    
    timestamps = np.array(timestamps)
    times,toronto = np.unique(torontoraw,return_counts=True)
    rtimes,torontor = np.unique(torontorraw,return_counts=True)
    htimes,torontoh = np.unique(torontohraw,return_counts=True)
    ftimes,torontof = np.unique(torontofraw,return_counts=True)
    torontor = np.array(matchtimes(times,rtimes,torontor))
    torontoh = np.array(matchtimes(times,htimes,torontoh))
    torontof = np.array(matchtimes(times,ftimes,torontof))
    TOneighborhoods["casesTO"] = toronto
    TOneighborhoods["fatalTO"] = torontof
    TOneighborhoods["hosptTO"] = torontoh
    TOneighborhoods["recovTO"] = torontor
    for neighborhood in TOneighborhoods["units"]:
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["CASES"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["CASES"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["CASES"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["FATAL"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["FATAL"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["FATAL"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["HOSPITALIZED"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["HOSPITALIZED"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["HOSPITALIZED"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["RECOVERED"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["RECOVERED"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["RECOVERED"] = np.array(matchtimes(times,time,cases))

    popfile = open("neighbourhood-profiles-2016-csv.csv","r")
    popcsv = csv.reader(popfile,delimiter=',',quotechar='"',quoting=csv.QUOTE_ALL,skipinitialspace=True)
    n = 0
    for row in popcsv:
        if n==0:
            header = row[6:]
        elif n==3:
            poprow = row[6:]
            break
        n+=1
        
    neighborhoodpops = {}
    for n in range(len(header)):
        neighborhoodpops[header[n]] = float(poprow[n].replace(",",""))
    #neighborhoodpops.keys()

    if "" in TOneighborhoods["units"]:
        TOneighborhoods["units"].pop("")
        
    keys1 = sorted(neighborhoodpops.keys())
    keys2 = sorted(TOneighborhoods["units"].keys())
    #print(len(keys1),len(keys2))
    for n in range(len(keys1)):
        #print("%40s\t%s\t%s"%(keys1[n],str(keys1[n]==keys2[n]),keys2[n]))
        if keys1[n]!=keys2[n]:
            neighborhoodpops[keys2[n]] = neighborhoodpops[keys1[n]]
            neighborhoodpops.pop(keys1[n])
            
    for neighborhood in neighborhoodpops:
        TOneighborhoods["units"][neighborhood]["POP"] = neighborhoodpops[neighborhood]
            
    otimes = {}
    ontario = {}
    ontario_d = {}
    ontario_a = {}
    ontario_r = {}
    with open("ontario_cases.csv","r") as df:
        ontariocsv =df.read().split("\n")[1:]
        while ontariocsv[-1]=="":
            ontariocsv = ontariocsv[:-1]
    for line in ontariocsv:
        entry = line.split(',')
        if entry[1]=="":
            entry[1]="UNKNOWN"
        if entry[1][0]=='"':
            entry[1] = entry[1][1:]
        if entry[1][-1]=='"':
            entry[1] = entry[1][:-1]
        phu = entry[1]
        i0 = 0
        if entry[0][0]=='"':
            entry[0]=entry[0][1:]
        if entry[0][-1]=='"':
            entry[0]=entry[0][:-1]
        timestamp = entry[0].split('-')
        try:
            x = timestamp[1]
        except:
            timestamp = timestamp[0]
            timestamp = [timestamp[:4],timestamp[4:6],timestamp[6:]]
        if phu=='HALIBURTON' or phu=='KINGSTON':
            phu=entry[1]+','+entry[2]+','+entry[3]
            i0 = 2
        elif phu=='LEEDS':
            phu=entry[1]+','+entry[2]
            i0 = 1
        phu = phu.replace('"','')
        active = int(entry[3+i0])
        rec    = int(entry[4+i0])
        dead   = int(entry[5+i0])
        if phu not in otimes.keys():
            otimes[phu] = np.array([date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])),])
            ontario_a[phu] = np.array([active,])
            ontario_r[phu] = np.array([rec,])
            ontario_d[phu] = np.array([dead,])
        else:
            try:
                otimes[phu] = np.append(otimes[phu],date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])))
                ontario_a[phu] = np.append(ontario_a[phu],active)
                ontario_r[phu] = np.append(ontario_r[phu],rec)
                ontario_d[phu] = np.append(ontario_d[phu],dead)
            except:
                print(timestamp)
    for phu in otimes.keys():
        order = np.argsort(otimes[phu])
        otimes[phu] = otimes[phu][order]
        ontario_a[phu] = ontario_a[phu][order]
        ontario_r[phu] = ontario_r[phu][order]
        ontario_d[phu] = ontario_d[phu][order]
        ontario[phu] = (np.diff(np.append([0,],ontario_a[phu]))
                       +np.diff(np.append([0,],ontario_r[phu]))
                       +np.diff(np.append([0,],ontario_d[phu]))) 
                     #ontario_a[phu]+ontario_d[phu]+ontario_r[phu]

    latestON = otimes["TORONTO"].max() #datetime.date
    latestTO = timestamps.max()        #datetime.date
    latestusa = usacsv[0][-1]
    usatime = latestusa.split("/")
    latestusa = date(2000+int(usatime[2]),int(usatime[0]),int(usatime[1]))
    globaltime = latestglobal.split("/")
    latestglobal = date(2000+int(globaltime[2]),int(globaltime[0]),int(globaltime[1]))
        
    TOneighborhoods["ProvincialTO"] = {"CASES":ontario["TORONTO"],
                                       "FATAL":ontario_d["TORONTO"],
                                       "ACTIVE":ontario_a["TORONTO"],
                                       "RECOVERED":ontario_r["TORONTO"]}
    
    time = ncd.createDimension("time",None)
    scalar = ncd.createDimension("scalar",1)
    
    ncd.set_auto_mask(False)
    
    nr = 100
    
    rdim = ncd.createDimension("Rt",nr)
    rrange = ncd.createVariable("rtrange","f8",("Rt",),zlib=True)
    rrange[:] = np.geomspace(0.1,10.0,num=nr)
    rbase = np.geomspace(0.01,10.0,num=1000)
    
    try:
        
        for country in sorted(countries):
            cdata = extract_country(dataset,country)
            if cdata["Total"][-1]>=25 and "Princess" not in country and "Olympics" not in country and "Zaandam" not in country:
                key = country
                if key=="US":
                    key="United States"
                ckey = str.title(key)
                countrygrp = ncd.createGroup(ckey)
                countrycases = ncd[ckey].createVariable("cases","i4",("time",),zlib=True)
                countrydeaths = ncd[ckey].createVariable("deaths","i4",("time",),zlib=True)
                countryRt = ncd[ckey].createVariable("Rt","f4",("time",),zlib=True)
                countryRpost = ncd[ckey].createVariable("Rpost","f8",("time","Rt"),zlib=True)
                countryRlike = ncd[ckey].createVariable("Rlike","f8",("time","Rt"),zlib=True)
                countrypopulation = ncd[ckey].createVariable("population","f4",("scalar",),zlib=True)
                ncd[ckey]["population"][:] = float(countrypops[country])
                ctotal = np.diff(np.append([0,],extract_country(dataset,country)["Total"])).astype(int)
                dtotal = np.diff(np.append([0,],extract_country(ddataset,country)["Total"])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal))
                p = np.exp(lp)
                l = np.exp(ll)
                pd = np.zeros((p.shape[0],nr))
                ld = np.zeros((l.shape[0],nr))
                for t in range(p.shape[0]):
                    pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                    ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                ncd[ckey]["Rt"][:] = r
                ncd[ckey]["Rpost"][:] = pd
                ncd[ckey]["Rlike"][:] = ld
                ncd[ckey]["cases"][:] = ctotal
                ncd[ckey]["deaths"][:] = dtotal
                countrycases.set_auto_mask(False)
                countrydeaths.set_auto_mask(False)
                countryRt.set_auto_mask(False)
                countrypopulation.set_auto_mask(False)
                countrycases.units = "cases day-1"
                countrydeaths.units = "deaths day-1"
                countryRt.units = "secondary infections case-1"
                countrypopulation.units = "people"
                countrycases.standard_name = "daily_cases"
                countrydeaths.standard_name = "daily_deaths"
                countryRt.standard_name = "R_t"
                countrypopulation.standard_name = "population"
                countrycases.long_name = "New Cases per Day"
                countrydeaths.long_name = "New Deaths per Day"
                countryRt.long_name = "Effective Reproductive Number"
                countrypopulation.long_name = "Population"
                
                countryRpost.units = "n/a"
                countryRlike.units = "n/a"
                countryRpost.set_auto_mask(False)
                countryRlike.set_auto_mask(False)
                countryRpost.standard_name = "R_t_posterior"
                countryRlike.standard_name = "R_t_likelihood"
                countryRpost.long_name = "Rt Posterior Probability"
                countryRpost.long_name = "Rt Likelihood Function"
                
        ncd.sync()
        
        for state in sorted(usa):
            if us_deaths[state][-1]>=20 and state!="Total" and "Princess" not in state\
                and "Virgin Islands" not in state and "Military" not in state\
                and "Recovered" not in state and "Prisons" not in state\
                and "Hospitals" not in state and state != "Total" and usa[state][-1]>150:
                    ckey = str.title(state)
                    stategrp = ncd["United States"].createGroup(ckey)
                    statecases = ncd["United States"][ckey].createVariable("cases","i2",("time",),zlib=True)
                    statedeaths = ncd["United States"][ckey].createVariable("deaths","i2",("time",),zlib=True)
                    stateRt = ncd["United States"][ckey].createVariable("Rt","f4",("time",),zlib=True)
                    stateRpost = ncd["United States"][ckey].createVariable("Rpost","f8",("time","Rt"),zlib=True)
                    stateRlike = ncd["United States"][ckey].createVariable("Rlike","f8",("time","Rt"),zlib=True)
                    statepopulation = ncd["United States"][ckey].createVariable("population","f4",("scalar",),zlib=True)
                    ncd["United States"][ckey]["population"][:] = float(statepops[state])
                    ctotal = np.diff(np.append([0,],usa[state])).astype(int)
                    dtotal = np.diff(np.append([0,],usa[state])).astype(int)
                    r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                    p = np.exp(lp)
                    l = np.exp(ll)
                    pd = np.zeros((p.shape[0],nr))
                    ld = np.zeros((l.shape[0],nr))
                    for t in range(p.shape[0]):
                        pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                        ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                    ncd["United States"][ckey]["Rt"][:] = r
                    ncd["United States"][ckey]["Rpost"][:] = pd
                    ncd["United States"][ckey]["Rlike"][:] = ld
                    ncd["United States"][ckey]["cases"][:] = ctotal
                    ncd["United States"][ckey]["deaths"][:] = dtotal
                    statecases.set_auto_mask(False)
                    statedeaths.set_auto_mask(False)
                    statepopulation.set_auto_mask(False)
                    statecases.units = "cases day-1"
                    statedeaths.units = "deaths day-1"
                    statepopulation.units = "people"
                    statecases.standard_name = "daily_cases"
                    statedeaths.standard_name = "daily_deaths"
                    statepopulation.standard_name = "population"
                    statecases.long_name = "new cases per day"
                    statedeaths.long_name = "new deaths per day"
                    statepopulation.long_name = "population"
                    stateRpost.units = "n/a"
                    stateRlike.units = "n/a"
                    stateRpost.set_auto_mask(False)
                    stateRlike.set_auto_mask(False)
                    stateRpost.standard_name = "R_t_posterior"
                    stateRlike.standard_name = "R_t_likelihood"
                    stateRpost.long_name = "Rt Posterior Probability"
                    stateRpost.long_name = "Rt Likelihood Function"
        ncd.sync()
        
        for province in sorted(canada):
            if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total":
                ckey = str.title(province)
                ckey = ckey.replace("And","and")
                provincegrp = ncd["Canada"].createGroup(ckey)
                provincecases = ncd["Canada"][ckey].createVariable("cases","i2",("time",),zlib=True)
                provincedeaths = ncd["Canada"][ckey].createVariable("deaths","i2",("time",),zlib=True)
                provinceRt = ncd["Canada"][ckey].createVariable("Rt","f4",("time",),zlib=True)
                provinceRpost = ncd["Canada"][ckey].createVariable("Rpost","f8",("time","Rt"),zlib=True)
                provinceRlike = ncd["Canada"][ckey].createVariable("Rlike","f8",("time","Rt"),zlib=True)
                provincepopulation = ncd["Canada"][ckey].createVariable("population","f4",("scalar",),zlib=True)
                ncd["Canada"][ckey]["population"][:] = float(provincepops[province])
                ctotal = np.diff(np.append([0,],canada[province])).astype(int)
                dtotal = np.diff(np.append([0,],canada[province])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                p = np.exp(lp)
                l = np.exp(ll)
                pd = np.zeros((p.shape[0],nr))
                ld = np.zeros((l.shape[0],nr))
                for t in range(p.shape[0]):
                    pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                    ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                ncd["Canada"][ckey]["Rt"][:] = r
                ncd["Canada"][ckey]["Rpost"][:] = pd
                ncd["Canada"][ckey]["Rlike"][:] = ld
                ncd["Canada"][ckey]["cases"][:] = ctotal
                ncd["Canada"][ckey]["deaths"][:] = dtotal
                provincecases.set_auto_mask(False)
                provincedeaths.set_auto_mask(False)
                provincepopulation.set_auto_mask(False)
                provincecases.units = "cases day-1"
                provincedeaths.units = "deaths day-1"
                provincepopulation.units = "people"
                provincecases.standard_name = "daily_cases"
                provincedeaths.standard_name = "daily_deaths"
                provincepopulation.standard_name = "population"
                provincecases.long_name = "new cases per day"
                provincedeaths.long_name = "new deaths per day"
                provincepopulation.long_name = "population"
                provinceRpost.units = "n/a"
                provinceRlike.units = "n/a"
                provinceRpost.set_auto_mask(False)
                provinceRlike.set_auto_mask(False)
                provinceRpost.standard_name = "R_t_posterior"
                provinceRlike.standard_name = "R_t_likelihood"
                provinceRpost.long_name = "Rt Posterior Probability"
                provinceRpost.long_name = "Rt Likelihood Function"
        
        ncd.sync()
        
        for phu in sorted(ontario):
            if len(ontario[phu][:])>10:
                ckey = str.title(phu)
                phugrp = ncd["Canada"]["Ontario"].createGroup(ckey)
                phucases = ncd["Canada/Ontario"][ckey].createVariable("cases","i2",("time",),zlib=True)
                phudeaths = ncd["Canada/Ontario"][ckey].createVariable("deaths","i2",("time",),zlib=True)
                phuactive = ncd["Canada/Ontario"][ckey].createVariable("active","i2",("time",),zlib=True)
                phurecovered = ncd["Canada/Ontario"][ckey].createVariable("recovered","i2",("time",),zlib=True)
                phuRt = ncd["Canada/Ontario"][ckey].createVariable("Rt","f4",("time",),zlib=True)
                phuRpost = ncd["Canada/Ontario"][ckey].createVariable("Rpost","f8",("time","Rt"),zlib=True)
                phuRlike = ncd["Canada/Ontario"][ckey].createVariable("Rlike","f8",("time","Rt"),zlib=True)
                #phupopulation = ncd["Canada/Ontario"][ckey].createVariable("population","f4",("scalar",),zlib=True)
                #ncd["Canada/Ontario"][ckey]["population"][:] = float(phupops[phu])
                ctotal = ontario[phu].astype(int)
                dtotal = np.diff(np.append([0,],ontario_d[phu])).astype(int)
                atotal = ontario_a[phu].astype(int)
                rtotal = np.diff(np.append([0,],ontario_r[phu])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                p = np.exp(lp)
                l = np.exp(ll)
                pd = np.zeros((p.shape[0],nr))
                ld = np.zeros((l.shape[0],nr))
                for t in range(p.shape[0]):
                    pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                    ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                ncd["Canada/Ontario"][ckey]["Rt"][:] = r
                ncd["Canada/Ontario"][ckey]["Rpost"][:] = pd
                ncd["Canada/Ontario"][ckey]["Rlike"][:] = ld
                ncd["Canada/Ontario"][ckey]["cases"][:] = ctotal
                ncd["Canada/Ontario"][ckey]["deaths"][:] = dtotal
                ncd["Canada/Ontario"][ckey]["active"][:] = atotal
                ncd["Canada/Ontario"][ckey]["recovered"][:] = rtotal
                phucases.set_auto_mask(False)
                phudeaths.set_auto_mask(False)
                phuactive.set_auto_mask(False)
                phurecovered.set_auto_mask(False)
                #phupopulation.set_auto_mask(False)
                phucases.units = "cases day-1"
                phudeaths.units = "deaths day-1"
                phuactive.units = "cases"
                phurecovered.units = "recoveries day-1"
                #phupopulation.units = "people"
                phucases.standard_name = "daily_cases"
                phudeaths.standard_name = "daily_deaths"
                phuactive.standard_name = "active_cases"
                phurecovered.standard_name = "daily_recoveries"
                #phupopulation.standard_name = "population"
                phucases.long_name = "new cases per day"
                phudeaths.long_name = "new deaths per day"
                phuactive.long_name = "active cases"
                phurecovered.long_name = "newly-recovered cases per day"
                #phupopulation.long_name = "population"
                phuRpost.units = "n/a"
                phuRlike.units = "n/a"
                phuRpost.set_auto_mask(False)
                phuRlike.set_auto_mask(False)
                phuRpost.standard_name = "R_t_posterior"
                phuRlike.standard_name = "R_t_likelihood"
                phuRpost.long_name = "Rt Posterior Probability"
                phuRpost.long_name = "Rt Likelihood Function"
            
        TOpop = ncd["Canada/Ontario/Toronto"].createVariable("population","f4",("scalar",),zlib=True)
        TOpop[:] = 2.732e6
        TOpop.set_auto_mask(False)
        TOpop.units = "people"
        TOpop.standard_name="population"
        TOpop.long_name="population"
        
        TOhosp = ncd["Canada/Ontario/Toronto"].createVariable("hospitalized","i2",("time",),zlib=True)
        TOhosp[:] = TOneighborhoods["hosptTO"].astype(int)
        TOhosp.set_auto_mask(False)
        TOhosp.units = "hospitalizations day-1"
        TOhosp.standard_name = "daily_hospitalizations"
        TOhosp.long_name = "day's cases which ever required hospitalization"
        
        ncd.sync()
        
        for neighborhood in sorted(TOneighborhoods["units"]):
            ckey = neighborhood.replace("/","-")
            areagrp = ncd["Canada/Ontario/Toronto"].createGroup(ckey)
            areacases = ncd["Canada/Ontario/Toronto"][ckey].createVariable("cases","i2",("time",),zlib=True)
            areadeaths = ncd["Canada/Ontario/Toronto"][ckey].createVariable("deaths","i2",("time",),zlib=True)
            areahosp = ncd["Canada/Ontario/Toronto"][ckey].createVariable("hospitalized","i2",("time",),zlib=True)
            arearecovered = ncd["Canada/Ontario/Toronto"][ckey].createVariable("recovered","i2",("time",),zlib=True)
            areaRt = ncd["Canada/Ontario/Toronto"][ckey].createVariable("Rt","f4",("time",),zlib=True)
            areaRpost = ncd["Canada/Ontario/Toronto"][ckey].createVariable("Rpost","f8",("time","Rt"),zlib=True)
            areaRlike = ncd["Canada/Ontario/Toronto"][ckey].createVariable("Rlike","f8",("time","Rt"),zlib=True)
            areapopulation = ncd["Canada/Ontario/Toronto"][ckey].createVariable("population","f4",("scalar",),zlib=True)
            ncd["Canada/Ontario/Toronto"][ckey]["population"][:] = float(TOneighborhoods["units"][neighborhood]["POP"])
            ctotal = TOneighborhoods["units"][neighborhood]["CASES"].astype(int)
            dtotal = TOneighborhoods["units"][neighborhood]["FATAL"].astype(int)
            htotal = TOneighborhoods["units"][neighborhood]["HOSPITALIZED"].astype(int)
            rtotal = TOneighborhoods["units"][neighborhood]["RECOVERED"].astype(int)
            r,lp,ll = Rt(day5avg(ctotal.astype(float)))
            p = np.exp(lp)
            l = np.exp(ll)
            pd = np.zeros((p.shape[0],nr))
            ld = np.zeros((l.shape[0],nr))
            for t in range(p.shape[0]):
                pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
            ncd["Canada/Ontario/Toronto"][ckey]["Rt"][:] = r
            ncd["Canada/Ontario/Toronto"][ckey]["Rpost"][:] = pd
            ncd["Canada/Ontario/Toronto"][ckey]["Rlike"][:] = ld
            ncd["Canada/Ontario/Toronto"][ckey]["cases"][:] = ctotal
            ncd["Canada/Ontario/Toronto"][ckey]["deaths"][:] = dtotal
            ncd["Canada/Ontario/Toronto"][ckey]["hospitalized"][:] = htotal
            ncd["Canada/Ontario/Toronto"][ckey]["recovered"][:] = rtotal
            areacases.set_auto_mask(False)
            areadeaths.set_auto_mask(False)
            areahosp.set_auto_mask(False)
            arearecovered.set_auto_mask(False)
            areapopulation.set_auto_mask(False)
            areacases.units = "cases day-1"
            areadeaths.units = "deaths day-1"
            areahosp.units = "hospitalizations day-1"
            arearecovered.units = "recoveries day-1"
            areapopulation.units = "people"
            areacases.standard_name = "daily_cases"
            areadeaths.standard_name = "daily_deaths"
            areahosp.standard_name = "daily_hospitalizations"
            arearecovered.standard_name = "daily_recoveries"
            areapopulation.standard_name = "population"
            areacases.long_name = "new cases per day"
            areadeaths.long_name = "new deaths per day"
            areahosp.long_name = "day's cases which ever required hospitalization"
            arearecovered.long_name = "day's cases which have recovered"
            areapopulation.long_name = "population"
            areaRpost.units = "n/a"
            areaRlike.units = "n/a"
            areaRpost.set_auto_mask(False)
            areaRlike.set_auto_mask(False)
            areaRpost.standard_name = "R_t_posterior"
            areaRlike.standard_name = "R_t_likelihood"
            areaRpost.long_name = "Rt Posterior Probability"
            areaRpost.long_name = "Rt Likelihood Function"
            
        ncd.sync()
        
        for state in sorted(usa):
            if us_deaths[state][-1]>=20 and state!="Total" and "Princess" not in state\
                and "Virgin Islands" not in state and "Military" not in state\
                and "Recovered" not in state and "Prisons" not in state and state!="Guam"\
                and "Hospitals" not in state and state != "Total" and usa[state][-1]>150:
                
                counties = get_counties(usacsv,state=state)
                for county in counties:
                    ckey = county.replace("/","-")
                    countygrp = ncd["United States/%s"%str.title(state)].createGroup(ckey)
                    countycases = ncd["United States/%s"%str.title(state)][ckey].createVariable("cases","i2",("time",),zlib=True)
                    countyRt = ncd["United States/%s"%str.title(state)][ckey].createVariable("Rt","f4",("time",),zlib=True)
                    countyRpost = ncd["United States/%s"%str.title(state)][ckey].createVariable("Rpost","f8",("time","Rt"),zlib=True)
                    countyRlike = ncd["United States/%s"%str.title(state)][ckey].createVariable("Rlike","f8",("time","Rt"),zlib=True)
                    countypopulation = ncd["United States/%s"%str.title(state)][ckey].createVariable("population","f4",("scalar",),zlib=True)
                    ncd["United States/%s"%str.title(state)][ckey]["population"][:] = float(get_countypop(county,state))
                    ctotal = np.diff(np.append([0,],extract_county(usacsv,county,state=state))).astype(int)
                    r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                    p = np.exp(lp)
                    l = np.exp(ll)
                    pd = np.zeros((p.shape[0],nr))
                    ld = np.zeros((l.shape[0],nr))
                    for t in range(p.shape[0]):
                        pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                        ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                    ncd["United States/%s"%str.title(state)][ckey]["Rt"][:] = r
                    ncd["United States/%s"%str.title(state)][ckey]["Rpost"][:] = pd
                    ncd["United States/%s"%str.title(state)][ckey]["Rlike"][:] = ld
                    ncd["United States/%s"%str.title(state)][ckey]["cases"][:] = ctotal
                    countycases.set_auto_mask(False)
                    countypopulation.set_auto_mask(False)
                    countycases.units = "cases day-1"
                    countypopulation.units = "people"
                    countycases.standard_name = "daily_cases"
                    countypopulation.standard_name = "population"
                    countycases.long_name = "New Cases per Day"
                    countypopulation.long_name = "Population"
                    countyRpost.units = "n/a"
                    countyRlike.units = "n/a"
                    countyRpost.set_auto_mask(False)
                    countyRlike.set_auto_mask(False)
                    countyRpost.standard_name = "R_t_posterior"
                    countyRlike.standard_name = "R_t_likelihood"
                    countyRpost.long_name = "Rt Posterior Probability"
                    countyRpost.long_name = "Rt Likelihood Function"
                    
                ncd.sync()
    except:
        ncd.close()
        raise
    ncd.close()
        
        
def hdf5():
    import h5py as h5
    
    countrysetf = "countrypopulations.csv"
    with open(countrysetf,"r") as df:
        countryset = df.read().split('\n')[1:]
    if countryset[-1] == "":
        countryset = countryset[:-1]
    countrypops = {}
    for line in countryset:
        linedata = line.split(',')
        name = linedata[0]
        popx = float(linedata[4])
        countrypops[name] = popx
    
    canadasetf = "provincepopulations.csv"
    with open(canadasetf,"r") as df:
        canadaset = df.read().split('\n')[2:]
    if canadaset[-1] == "":
        canadaset = canadaset[:-1]
    provincepops = {}
    for line in canadaset:
        linedata = line.split(',')
        name = linedata[1]
        popx = float(linedata[-3])
        provincepops[name] = popx
        
    statesetf = "statepopulations.csv"
    with open(statesetf,"r") as df:
        stateset = df.read().split('\n')[2:]
    if stateset[-1] == "":
        stateset = stateset[:-1]
    statepops = {}
    for line in stateset:
        linedata = line.split(',')
        name = linedata[2]
        popx = float(linedata[3])
        statepops[name] = popx
        
    #_log("/home/adivp416/public_html/covid19/reportlog.txt","Static CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
        
    globalconf = "github/time_series_covid19_confirmed_global.csv"

    with open(globalconf,"r") as df:
        dataset = df.read().split('\n')
    header = dataset[0].split(',')
    dataset = dataset[1:]
    if dataset[-1]=='':
        dataset = dataset[:-1]
    latestglobal = header[-1]
        
    countries = getcountries(dataset)
    

    usacsv = []
    with open("github/time_series_covid19_confirmed_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usacsv.append(row)
            
    usa = extract_usa(usacsv)

    canada = extract_country(dataset,"Canada")

    ddatasetf = "github/time_series_covid19_deaths_global.csv"

    with open(ddatasetf,"r") as df:
        ddataset = df.read().split('\n')
    header = ddataset[0].split(',')
    ddataset = ddataset[1:]
    if ddataset[-1]=='':
        ddataset = ddataset[:-1]
        
    usadcsv = []
    with open("github/time_series_covid19_deaths_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usadcsv.append(row)
            
    #_log("/home/adivp416/public_html/covid19/reportlog.txt","Dynamic CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
    
    ca_deaths = extract_country(ddataset,"Canada")
    us_deaths = extract_usa(usadcsv)

    timestamps = []
    rtimes = []
    ftimes = []
    htimes = []
    TOneighborhoods = {"units":{}}
    with open("toronto_cases.csv","r") as df:
        torontocsv = df.read().split('\n')[1:]
        while torontocsv[-1]=="":
            torontocsv = torontocsv[:-1]
        torontoraw = np.zeros(len(torontocsv))
        for line in range(len(torontoraw)):
            timestamp = torontocsv[line].split(',')[9].split('-')
            timestamps.append(date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])))
            entry = torontocsv[line].split(',')
            if entry[11]=="ACTIVE":
                pass
            elif entry[11]=="FATAL":
                ftimes.append(timestamps[-1])
            else:
                rtimes.append(timestamps[-1])
            if entry[15]=="Yes":
                htimes.append(timestamps[-1])
                
            if entry[4] not in TOneighborhoods["units"]:
                TOneighborhoods["units"][entry[4]] = {"CASES":[],"FATAL":[],"HOSPITALIZED":[],"RECOVERED":[]}
            TOneighborhoods["units"][entry[4]]["CASES"].append(timestamps[-1])
            if entry[11]=="ACTIVE":
                pass
            elif entry[11]=="FATAL":
                TOneighborhoods["units"][entry[4]]["FATAL"].append(timestamps[-1])
            else:
                TOneighborhoods["units"][entry[4]]["RECOVERED"].append(timestamps[-1])
            if entry[15]=="Yes":
                TOneighborhoods["units"][entry[4]]["HOSPITALIZED"].append(timestamps[-1])
            
    #for neighborhood in TOneighborhoods["units"]:
    #    TOneighborhoods["units"][neighborhood]["CASES"] = np.array(TOneighborhoods["units"][neighborhood]["CASES"])
    #    TOneighborhoods["units"][neighborhood]["FATAL"] = np.array(TOneighborhoods["units"][neighborhood]["FATAL"])
    #    TOneighborhoods["units"][neighborhood]["HOSPITALIZED"] = np.array(TOneighborhoods["units"][neighborhood]["HOSPITALIZED"])
    #    TOneighborhoods["units"][neighborhood]["RECOVERED"] = np.array(TOneighborhoods["units"][neighborhood]["RECOVERED"])
            
    #All cases
    origin = np.array(timestamps).min()
    for line in range(len(timestamps)):
        day = ((timestamps[line]-origin).days)
        torontoraw[line] = int(day)
    #Recovered cases
    torontorraw = np.zeros(len(rtimes))
    for line in range(len(rtimes)):
        day = ((rtimes[line]-origin).days)
        torontorraw[line] = int(day)
    #Hospitalized cases
    torontohraw = np.zeros(len(htimes))
    for line in range(len(htimes)):
        day = ((htimes[line]-origin).days)
        torontohraw[line] = int(day)
    #Fatal cases
    torontofraw = np.zeros(len(ftimes))
    for line in range(len(ftimes)):
        day = ((ftimes[line]-origin).days)
        torontofraw[line] = int(day)
    
    timestamps = np.array(timestamps)
    times,toronto = np.unique(torontoraw,return_counts=True)
    rtimes,torontor = np.unique(torontorraw,return_counts=True)
    htimes,torontoh = np.unique(torontohraw,return_counts=True)
    ftimes,torontof = np.unique(torontofraw,return_counts=True)
    torontor = np.array(matchtimes(times,rtimes,torontor))
    torontoh = np.array(matchtimes(times,htimes,torontoh))
    torontof = np.array(matchtimes(times,ftimes,torontof))
    TOneighborhoods["casesTO"] = toronto
    TOneighborhoods["fatalTO"] = torontof
    TOneighborhoods["hosptTO"] = torontoh
    TOneighborhoods["recovTO"] = torontor
    for neighborhood in TOneighborhoods["units"]:
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["CASES"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["CASES"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["CASES"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["FATAL"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["FATAL"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["FATAL"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["HOSPITALIZED"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["HOSPITALIZED"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["HOSPITALIZED"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["RECOVERED"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["RECOVERED"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["RECOVERED"] = np.array(matchtimes(times,time,cases))

    popfile = open("neighbourhood-profiles-2016-csv.csv","r")
    popcsv = csv.reader(popfile,delimiter=',',quotechar='"',quoting=csv.QUOTE_ALL,skipinitialspace=True)
    n = 0
    for row in popcsv:
        if n==0:
            header = row[6:]
        elif n==3:
            poprow = row[6:]
            break
        n+=1
        
    neighborhoodpops = {}
    for n in range(len(header)):
        neighborhoodpops[header[n]] = float(poprow[n].replace(",",""))
    #neighborhoodpops.keys()

    if "" in TOneighborhoods["units"]:
        TOneighborhoods["units"].pop("")
        
    keys1 = sorted(neighborhoodpops.keys())
    keys2 = sorted(TOneighborhoods["units"].keys())
    #print(len(keys1),len(keys2))
    for n in range(len(keys1)):
        #print("%40s\t%s\t%s"%(keys1[n],str(keys1[n]==keys2[n]),keys2[n]))
        if keys1[n]!=keys2[n]:
            neighborhoodpops[keys2[n]] = neighborhoodpops[keys1[n]]
            neighborhoodpops.pop(keys1[n])
            
    for neighborhood in neighborhoodpops:
        TOneighborhoods["units"][neighborhood]["POP"] = neighborhoodpops[neighborhood]
            
    otimes = {}
    ontario = {}
    ontario_d = {}
    ontario_a = {}
    ontario_r = {}
    with open("ontario_cases.csv","r") as df:
        ontariocsv =df.read().split("\n")[1:]
        while ontariocsv[-1]=="":
            ontariocsv = ontariocsv[:-1]
    for line in ontariocsv:
        entry = line.split(',')
        if entry[1]=="":
            entry[1]="UNKNOWN"
        if entry[1][0]=='"':
            entry[1] = entry[1][1:]
        if entry[1][-1]=='"':
            entry[1] = entry[1][:-1]
        phu = entry[1]
        i0 = 0
        if entry[0][0]=='"':
            entry[0]=entry[0][1:]
        if entry[0][-1]=='"':
            entry[0]=entry[0][:-1]
        timestamp = entry[0].split('-')
        try:
            x = timestamp[1]
        except:
            timestamp = timestamp[0]
            timestamp = [timestamp[:4],timestamp[4:6],timestamp[6:]]
        if phu=='HALIBURTON' or phu=='KINGSTON':
            phu=entry[1]+','+entry[2]+','+entry[3]
            i0 = 2
        elif phu=='LEEDS':
            phu=entry[1]+','+entry[2]
            i0 = 1
        phu = phu.replace('"','')
        active = int(entry[3+i0])
        rec    = int(entry[4+i0])
        dead   = int(entry[5+i0])
        if phu not in otimes.keys():
            otimes[phu] = np.array([date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])),])
            ontario_a[phu] = np.array([active,])
            ontario_r[phu] = np.array([rec,])
            ontario_d[phu] = np.array([dead,])
        else:
            try:
                otimes[phu] = np.append(otimes[phu],date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])))
                ontario_a[phu] = np.append(ontario_a[phu],active)
                ontario_r[phu] = np.append(ontario_r[phu],rec)
                ontario_d[phu] = np.append(ontario_d[phu],dead)
            except:
                print(timestamp)
    for phu in otimes.keys():
        order = np.argsort(otimes[phu])
        otimes[phu] = otimes[phu][order]
        ontario_a[phu] = ontario_a[phu][order]
        ontario_r[phu] = ontario_r[phu][order]
        ontario_d[phu] = ontario_d[phu][order]
        ontario[phu] = (np.diff(np.append([0,],ontario_a[phu]))
                       +np.diff(np.append([0,],ontario_r[phu]))
                       +np.diff(np.append([0,],ontario_d[phu]))) 
                     #ontario_a[phu]+ontario_d[phu]+ontario_r[phu]

    latestON = otimes["TORONTO"].max() #datetime.date
    latestTO = timestamps.max()        #datetime.date
    latestusa = usacsv[0][-1]
    usatime = latestusa.split("/")
    latestusa = date(2000+int(usatime[2]),int(usatime[0]),int(usatime[1]))
    globaltime = latestglobal.split("/")
    latestglobal = date(2000+int(globaltime[2]),int(globaltime[0]),int(globaltime[1]))
    latestON = ' '.join([x for i,x in enumerate(latestON.ctime().split()) if i!=3])
    latestTO = ' '.join([x for i,x in enumerate(latestTO.ctime().split()) if i!=3])
    latestusa = ' '.join([x for i,x in enumerate(latestusa.ctime().split()) if i!=3])
    latestglobal = ' '.join([x for i,x in enumerate(latestglobal.ctime().split()) if i!=3])
        
    TOneighborhoods["ProvincialTO"] = {"CASES":ontario["TORONTO"],
                                       "FATAL":ontario_d["TORONTO"],
                                       "ACTIVE":ontario_a["TORONTO"],
                                       "RECOVERED":ontario_r["TORONTO"]}
    
    hdf = h5.File("adivparadise_covid19data.hdf5","w")
    
    nr = 100
    
    rrange = np.geomspace(0.1,10.0,num=nr)
    hdf.create_dataset("/rtrange",data=rrange.astype("float32"),compression='gzip',
                       compression_opts=9,shuffle=True,fletcher32=True)
    hdf["rtrange"].attrs['long_name'] = "Possible R$_t$ Values"
    rbase = np.geomspace(0.01,10.0,num=1000)
    
    print("doing countries")
    try:
        for country in sorted(countries):
            cdata = extract_country(dataset,country)
            if cdata["Total"][-1]>=25 and "Princess" not in country and "Olympics" not in country and "Zaandam" not in country:
                key = country
                if key=="US":
                    key="United States"
                ckey = str.title(key)
                ctotal = np.diff(np.append([0,],extract_country(dataset,country)["Total"])).astype(int)
                dtotal = np.diff(np.append([0,],extract_country(ddataset,country)["Total"])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal))
                p = np.exp(lp)
                l = np.exp(ll)
                pd = np.zeros((p.shape[0],nr))
                ld = np.zeros((l.shape[0],nr))
                for t in range(p.shape[0]):
                    pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                    ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                    
                countrycases = hdf.create_dataset("/"+ckey+"/cases",compression='gzip',compression_opts=9,
                                                  shuffle=True,fletcher32=True,data=ctotal)
                countrydeaths = hdf.create_dataset("/"+ckey+"/deaths",compression='gzip',compression_opts=9,
                                                  shuffle=True,fletcher32=True,data=dtotal)
                countryRt = hdf.create_dataset("/"+ckey+"/Rt",compression='gzip',compression_opts=9,
                                                  shuffle=True,fletcher32=True,data=r.astype("float32"))
                countryRpost = hdf.create_dataset("/"+ckey+"/Rpost",compression='gzip',compression_opts=9,
                                                  shuffle=True,fletcher32=True,data=pd)
                countryRlike = hdf.create_dataset("/"+ckey+"/Rlike",compression='gzip',compression_opts=9,
                                                  shuffle=True,fletcher32=True,data=ld)
                countrypopulation = hdf.create_dataset("/"+ckey+"/population",data=float(countrypops[country]))
                latest = hdf.create_dataset("/"+ckey+"/latestdate",data=latestglobal)

                countrycases.attrs["units"] = "cases day-1"
                countrydeaths.attrs["units"] = "deaths day-1"
                countryRt.attrs["units"] = "secondary infections case-1"
                countrypopulation.attrs["units"] = "people"
                countrycases.attrs["standard_name"] = "daily_cases"
                countrydeaths.attrs["standard_name"] = "daily_deaths"
                countryRt.attrs["standard_name"] = "R_t"
                countrypopulation.attrs["standard_name"] = "population"
                countrycases.attrs["long_name"] = "New Cases per Day"
                countrydeaths.attrs["long_name"] = "New Deaths per Day"
                countryRt.attrs["long_name"] = "Effective Reproductive Number"
                countrypopulation.attrs["long_name"] = "Population"
                
                countryRpost.attrs["units"] = "n/a"
                countryRlike.attrs["units"] = "n/a"
                countryRpost.attrs["standard_name"] = "R_t_posterior"
                countryRlike.attrs["standard_name"] = "R_t_likelihood"
                countryRpost.attrs["long_name"] = "Rt Posterior Probability"
                countryRpost.attrs["long_name"] = "Rt Likelihood Function"
        print("doing states")
        for state in sorted(usa):
            if us_deaths[state][-1]>=20 and state!="Total" and "Princess" not in state\
                and "Virgin Islands" not in state and "Military" not in state\
                and "Recovered" not in state and "Prisons" not in state\
                and "Hospitals" not in state and state != "Total" and usa[state][-1]>150:
                    ckey = str.title(state)
                    ctotal = np.diff(np.append([0,],usa[state])).astype(int)
                    dtotal = np.diff(np.append([0,],usa[state])).astype(int)
                    r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                    p = np.exp(lp)
                    l = np.exp(ll)
                    pd = np.zeros((p.shape[0],nr))
                    ld = np.zeros((l.shape[0],nr))
                    for t in range(p.shape[0]):
                        pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                        ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                        
                    statecases  = hdf.create_dataset("/United States/"+ckey+"/cases",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ctotal.astype(np.short))
                    statedeaths = hdf.create_dataset("/United States/"+ckey+"/deaths",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=dtotal.astype(np.short))
                    stateRt     = hdf.create_dataset("/United States/"+ckey+"/Rt",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=r.astype("float32"))
                    stateRpost  = hdf.create_dataset("/United States/"+ckey+"/Rpost",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=pd)
                    stateRlike  = hdf.create_dataset("/United States/"+ckey+"/Rlike",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ld)
                    statepopulation = hdf.create_dataset("/United States/"+ckey+"/population",data=float(statepops[state]))
                    latest = hdf.create_dataset("/United States/"+ckey+"/latestdate",data=latestusa)
  
                    statecases.attrs["units"] = "cases day-1"
                    statedeaths.attrs["units"] = "deaths day-1"
                    statepopulation.attrs["units"] = "people"
                    statecases.attrs["standard_name"] = "daily_cases"
                    statedeaths.attrs["standard_name"] = "daily_deaths"
                    statepopulation.attrs["standard_name"] = "population"
                    statecases.attrs["long_name"] = "new cases per day"
                    statedeaths.attrs["long_name"] = "new deaths per day"
                    statepopulation.attrs["long_name"] = "population"
                    stateRpost.attrs["units"] = "n/a"
                    stateRlike.attrs["units"] = "n/a"

                    stateRpost.attrs["standard_name"] = "R_t_posterior"
                    stateRlike.attrs["standard_name"] = "R_t_likelihood"
                    stateRpost.attrs["long_name"] = "Rt Posterior Probability"
                    stateRpost.attrs["long_name"] = "Rt Likelihood Function"     
        print("Doing provinces")   
        for province in sorted(canada):
            if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total":
                ckey = str.title(province)
                ckey = ckey.replace("And","and")
                ctotal = np.diff(np.append([0,],canada[province])).astype(int)
                dtotal = np.diff(np.append([0,],canada[province])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                p = np.exp(lp)
                l = np.exp(ll)
                pd = np.zeros((p.shape[0],nr))
                ld = np.zeros((l.shape[0],nr))
                for t in range(p.shape[0]):
                    pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                    ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                    
                provincecases  = hdf.create_dataset("/Canada/"+ckey+"/cases",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ctotal.astype(np.short))
                provincedeaths = hdf.create_dataset("/Canada/"+ckey+"/deaths",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=dtotal.astype(np.short))
                provinceRt     = hdf.create_dataset("/Canada/"+ckey+"/Rt",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=r.astype("float32"))
                provinceRpost  = hdf.create_dataset("/Canada/"+ckey+"/Rpost",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=pd)
                provinceRlike  = hdf.create_dataset("/Canada/"+ckey+"/Rlike",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ld)
                provincepopulation = hdf.create_dataset("/Canada/"+ckey+"/population",data=float(provincepops[province]))
                latest = hdf.create_dataset("/Canada/"+ckey+"/latestdate",data=latestglobal)

                provincecases.attrs["units"] = "cases day-1"
                provincedeaths.attrs["units"] = "deaths day-1"
                provincepopulation.attrs["units"] = "people"
                provincecases.attrs["standard_name"] = "daily_cases"
                provincedeaths.attrs["standard_name"] = "daily_deaths"
                provincepopulation.attrs["standard_name"] = "population"
                provincecases.attrs["long_name"] = "new cases per day"
                provincedeaths.attrs["long_name"] = "new deaths per day"
                provincepopulation.attrs["long_name"] = "population"
                provinceRpost.attrs["units"] = "n/a"
                provinceRlike.attrs["units"] = "n/a"

                provinceRpost.attrs["standard_name"] = "R_t_posterior"
                provinceRlike.attrs["standard_name"] = "R_t_likelihood"
                provinceRpost.attrs["long_name"] = "Rt Posterior Probability"
                provinceRpost.attrs["long_name"] = "Rt Likelihood Function"
        
        print("Doing PHUs")
        for phu in sorted(ontario):
            if len(ontario[phu][:])>10:
                ckey = str.title(phu)
                ctotal = ontario[phu].astype(int)
                dtotal = np.diff(np.append([0,],ontario_d[phu])).astype(int)
                atotal = ontario_a[phu].astype(int)
                rtotal = np.diff(np.append([0,],ontario_r[phu])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                p = np.exp(lp)
                l = np.exp(ll)
                pd = np.zeros((p.shape[0],nr))
                ld = np.zeros((l.shape[0],nr))
                for t in range(p.shape[0]):
                    pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                    ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                phucases = hdf.create_dataset("/Canada/Ontario/"+ckey+"/cases",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ctotal.astype(np.short))
                phudeaths = hdf.create_dataset("/Canada/Ontario/"+ckey+"/deaths",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=dtotal.astype(np.short))
                phuactive = hdf.create_dataset("/Canada/Ontario/"+ckey+"/active",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=atotal.astype(np.short))
                phurecovered = hdf.create_dataset("/Canada/Ontario/"+ckey+"/recovered",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=rtotal.astype(np.short))
                phuRt = hdf.create_dataset("/Canada/Ontario/"+ckey+"/Rt",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=r.astype('float32'))
                phuRpost = hdf.create_dataset("/Canada/Ontario/"+ckey+"/Rpost",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=pd)
                phuRlike = hdf.create_dataset("/Canada/Ontario/"+ckey+"/Rlike",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ld)
                latest = hdf.create_dataset("/Canada/Ontario/"+ckey+"/latestdate",data=latestON)
                #phupopulation = ncd["Canada/Ontario"][ckey].createVariable("population","f4",("scalar",),zlib=True)
                #ncd["Canada/Ontario"][ckey]["population"][:] = float(phupops[phu])
                #phupopulation.set_auto_mask(False)
                phucases.attrs["units"] = "cases day-1"
                phudeaths.attrs["units"] = "deaths day-1"
                phuactive.attrs["units"] = "cases"
                phurecovered.attrs["units"] = "recoveries day-1"
                #phupopulation.units = "people"
                phucases.attrs["standard_name"] = "daily_cases"
                phudeaths.attrs["standard_name"] = "daily_deaths"
                phuactive.attrs["standard_name"] = "active_cases"
                phurecovered.attrs["standard_name"] = "daily_recoveries"
                #phupopulation.standard_name = "population"
                phucases.attrs["long_name"] = "new cases per day"
                phudeaths.attrs["long_name"] = "new deaths per day"
                phuactive.attrs["long_name"] = "active cases"
                phurecovered.attrs["long_name"] = "newly-recovered cases per day"
                #phupopulation.long_name = "population"
                phuRpost.attrs["units"] = "n/a"
                phuRlike.attrs["units"] = "n/a"
                phuRpost.attrs["standard_name"] = "R_t_posterior"
                phuRlike.attrs["standard_name"] = "R_t_likelihood"
                phuRpost.attrs["long_name"] = "Rt Posterior Probability"
                phuRpost.attrs["long_name"] = "Rt Likelihood Function"
            
        TOpop = hdf.create_dataset("/Canada/Ontario/Toronto/population",data=2.732e6)

        TOpop.attrs["units"] = "people"
        TOpop.attrs["standard_name"]="population"
        TOpop.attrs["long_name"]="population"
        
        TOhosp = hdf.create_dataset("/Canada/Ontario/Toronto/hospitalized",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=TOneighborhoods["hosptTO"].astype(np.short))

        TOhosp.attrs["units"] = "hospitalizations day-1"
        TOhosp.attrs["standard_name"] = "daily_hospitalizations"
        TOhosp.attrs["long_name"] = "day's cases which ever required hospitalization"
        
        print("doing neighborhoods")
        for neighborhood in sorted(TOneighborhoods["units"]):
            ckey = neighborhood.replace("/","-")
            ctotal = TOneighborhoods["units"][neighborhood]["CASES"].astype(int)
            dtotal = TOneighborhoods["units"][neighborhood]["FATAL"].astype(int)
            htotal = TOneighborhoods["units"][neighborhood]["HOSPITALIZED"].astype(int)
            rtotal = TOneighborhoods["units"][neighborhood]["RECOVERED"].astype(int)
            r,lp,ll = Rt(day5avg(ctotal.astype(float)))
            p = np.exp(lp)
            l = np.exp(ll)
            pd = np.zeros((p.shape[0],nr))
            ld = np.zeros((l.shape[0],nr))
            for t in range(p.shape[0]):
                pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                ld[t,:] = np.interp(rrange[:],rbase,l[t,:])

            areacases = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/cases",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ctotal.astype(np.short))
            areadeaths = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/deaths",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=dtotal.astype(np.short))
            areahosp = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/hospitalized",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=htotal.astype(np.short))
            arearecovered = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/recovered",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=rtotal.astype(np.short))
            areaRt = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/Rt",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=r.astype("float32"))
            areaRpost = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/Rpost",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=pd)
            areaRlike = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/Rlike",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ld)
            areapopulation = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/population",
                                                data=float(TOneighborhoods["units"][neighborhood]["POP"]))
            latest = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/latestdate",data=latestTO)

            areacases.attrs["units"] = "cases day-1"
            areadeaths.attrs["units"] = "deaths day-1"
            areahosp.attrs["units"] = "hospitalizations day-1"
            arearecovered.attrs["units"] = "recoveries day-1"
            areapopulation.attrs["units"] = "people"
            areacases.attrs["standard_name"] = "daily_cases"
            areadeaths.attrs["standard_name"] = "daily_deaths"
            areahosp.attrs["standard_name"] = "daily_hospitalizations"
            arearecovered.attrs["standard_name"] = "daily_recoveries"
            areapopulation.attrs["standard_name"] = "population"
            areacases.attrs["long_name"] = "new cases per day"
            areadeaths.attrs["long_name"] = "new deaths per day"
            areahosp.attrs["long_name"] = "day's cases which ever required hospitalization"
            arearecovered.attrs["long_name"] = "day's cases which have recovered"
            areapopulation.attrs["long_name"] = "population"
            areaRpost.attrs["units"] = "n/a"
            areaRlike.attrs["units"] = "n/a"
            areaRpost.attrs["standard_name"] = "R_t_posterior"
            areaRlike.attrs["standard_name"] = "R_t_likelihood"
            areaRpost.attrs["long_name"] = "Rt Posterior Probability"
            areaRpost.attrs["long_name"] = "Rt Likelihood Function"
        print("Doing counties")          
        for state in sorted(usa):
            if us_deaths[state][-1]>=20 and state!="Total" and "Princess" not in state\
                and "Virgin Islands" not in state and "Military" not in state\
                and "Recovered" not in state and "Prisons" not in state and state!="Guam"\
                and "Hospitals" not in state and state != "Total" and usa[state][-1]>150:
                
                counties = get_counties(usacsv,state=state)
                for county in counties:
                    ckey = county.replace("/","-")
                    ctotal = np.diff(np.append([0,],extract_county(usacsv,county,state=state))).astype(int)
                    r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                    p = np.exp(lp)
                    l = np.exp(ll)
                    pd = np.zeros((p.shape[0],nr))
                    ld = np.zeros((l.shape[0],nr))
                    for t in range(p.shape[0]):
                        pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                        ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                        
                    countycases = hdf.create_dataset("/United States/%s"%str.title(state)+"/"+ckey+"/cases",
                                                     compression='gzip',compression_opts=9,shuffle=True,
                                                     fletcher32=True,data=ctotal.astype(np.short))
                    countyRt = hdf.create_dataset("/United States/%s"%str.title(state)+"/"+ckey+"/Rt",
                                                     compression='gzip',compression_opts=9,shuffle=True,
                                                     fletcher32=True,data=r.astype('float'))
                    countyRpost = hdf.create_dataset("/United States/%s"%str.title(state)+"/"+ckey+"/Rpost",
                                                     compression='gzip',compression_opts=9,shuffle=True,
                                                     fletcher32=True,data=pd)
                    countyRlike = hdf.create_dataset("/United States/%s"%str.title(state)+"/"+ckey+"/Rlike",
                                                     compression='gzip',compression_opts=9,shuffle=True,
                                                     fletcher32=True,data=ld)
                    countypopulation = hdf.create_dataset("/United States/%s"%str.title(state)+"/"+ckey+"/population",
                                                     data=float(get_countypop(county,state)))
                    latest = hdf.create_dataset("/United States/%s"%str.title(state)+"/"+ckey+"/latestdate",
                                                data=latestusa)

                    countycases.attrs["units"] = "cases day-1"
                    countypopulation.attrs["units"] = "people"
                    countycases.attrs["standard_name"] = "daily_cases"
                    countypopulation.attrs["standard_name"] = "population"
                    countycases.attrs["long_name"] = "New Cases per Day"
                    countypopulation.attrs["long_name"] = "Population"
                    countyRpost.attrs["units"] = "n/a"
                    countyRlike.attrs["units"] = "n/a"
                    countyRpost.attrs["standard_name"] = "R_t_posterior"
                    countyRlike.attrs["standard_name"] = "R_t_likelihood"
                    countyRpost.attrs["long_name"] = "Rt Posterior Probability"
                    countyRpost.attrs["long_name"] = "Rt Likelihood Function"
        
    except:
        hdf.close()
        raise
    
    hdf.close()
    print("File completed and closed")
 
def netcdf_slim():
    import netCDF4 as nc
    
    ncd = nc.Dataset("adivparadise_covid19data_slim.nc","w")
    
    countrysetf = "countrypopulations.csv"
    with open(countrysetf,"r") as df:
        countryset = df.read().split('\n')[1:]
    if countryset[-1] == "":
        countryset = countryset[:-1]
    countrypops = {}
    for line in countryset:
        linedata = line.split(',')
        name = linedata[0]
        popx = float(linedata[4])
        countrypops[name] = popx
    
    canadasetf = "provincepopulations.csv"
    with open(canadasetf,"r") as df:
        canadaset = df.read().split('\n')[2:]
    if canadaset[-1] == "":
        canadaset = canadaset[:-1]
    provincepops = {}
    for line in canadaset:
        linedata = line.split(',')
        name = linedata[1]
        popx = float(linedata[-3])
        provincepops[name] = popx
        
    statesetf = "statepopulations.csv"
    with open(statesetf,"r") as df:
        stateset = df.read().split('\n')[2:]
    if stateset[-1] == "":
        stateset = stateset[:-1]
    statepops = {}
    for line in stateset:
        linedata = line.split(',')
        name = linedata[2]
        popx = float(linedata[3])
        statepops[name] = popx
        
    #_log("/home/adivp416/public_html/covid19/reportlog.txt","Static CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
        
    globalconf = "github/time_series_covid19_confirmed_global.csv"

    with open(globalconf,"r") as df:
        dataset = df.read().split('\n')
    header = dataset[0].split(',')
    dataset = dataset[1:]
    if dataset[-1]=='':
        dataset = dataset[:-1]
    latestglobal = header[-1]
        
    countries = getcountries(dataset)
    

    usacsv = []
    with open("github/time_series_covid19_confirmed_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usacsv.append(row)
            
    usa = extract_usa(usacsv)

    canada = extract_country(dataset,"Canada")

    ddatasetf = "github/time_series_covid19_deaths_global.csv"

    with open(ddatasetf,"r") as df:
        ddataset = df.read().split('\n')
    header = ddataset[0].split(',')
    ddataset = ddataset[1:]
    if ddataset[-1]=='':
        ddataset = ddataset[:-1]
        
    usadcsv = []
    with open("github/time_series_covid19_deaths_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usadcsv.append(row)
            
    #_log("/home/adivp416/public_html/covid19/reportlog.txt","Dynamic CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
    
    ca_deaths = extract_country(ddataset,"Canada")
    us_deaths = extract_usa(usadcsv)

    timestamps = []
    rtimes = []
    ftimes = []
    htimes = []
    TOneighborhoods = {"units":{}}
    with open("toronto_cases.csv","r") as df:
        torontocsv = df.read().split('\n')[1:]
        while torontocsv[-1]=="":
            torontocsv = torontocsv[:-1]
        torontoraw = np.zeros(len(torontocsv))
        for line in range(len(torontoraw)):
            timestamp = torontocsv[line].split(',')[9].split('-')
            timestamps.append(date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])))
            entry = torontocsv[line].split(',')
            if entry[11]=="ACTIVE":
                pass
            elif entry[11]=="FATAL":
                ftimes.append(timestamps[-1])
            else:
                rtimes.append(timestamps[-1])
            if entry[15]=="Yes":
                htimes.append(timestamps[-1])
                
            if entry[4] not in TOneighborhoods["units"]:
                TOneighborhoods["units"][entry[4]] = {"CASES":[],"FATAL":[],"HOSPITALIZED":[],"RECOVERED":[]}
            TOneighborhoods["units"][entry[4]]["CASES"].append(timestamps[-1])
            if entry[11]=="ACTIVE":
                pass
            elif entry[11]=="FATAL":
                TOneighborhoods["units"][entry[4]]["FATAL"].append(timestamps[-1])
            else:
                TOneighborhoods["units"][entry[4]]["RECOVERED"].append(timestamps[-1])
            if entry[15]=="Yes":
                TOneighborhoods["units"][entry[4]]["HOSPITALIZED"].append(timestamps[-1])
            
    #for neighborhood in TOneighborhoods["units"]:
    #    TOneighborhoods["units"][neighborhood]["CASES"] = np.array(TOneighborhoods["units"][neighborhood]["CASES"])
    #    TOneighborhoods["units"][neighborhood]["FATAL"] = np.array(TOneighborhoods["units"][neighborhood]["FATAL"])
    #    TOneighborhoods["units"][neighborhood]["HOSPITALIZED"] = np.array(TOneighborhoods["units"][neighborhood]["HOSPITALIZED"])
    #    TOneighborhoods["units"][neighborhood]["RECOVERED"] = np.array(TOneighborhoods["units"][neighborhood]["RECOVERED"])
            
    #All cases
    origin = np.array(timestamps).min()
    for line in range(len(timestamps)):
        day = ((timestamps[line]-origin).days)
        torontoraw[line] = int(day)
    #Recovered cases
    torontorraw = np.zeros(len(rtimes))
    for line in range(len(rtimes)):
        day = ((rtimes[line]-origin).days)
        torontorraw[line] = int(day)
    #Hospitalized cases
    torontohraw = np.zeros(len(htimes))
    for line in range(len(htimes)):
        day = ((htimes[line]-origin).days)
        torontohraw[line] = int(day)
    #Fatal cases
    torontofraw = np.zeros(len(ftimes))
    for line in range(len(ftimes)):
        day = ((ftimes[line]-origin).days)
        torontofraw[line] = int(day)
    
    timestamps = np.array(timestamps)
    times,toronto = np.unique(torontoraw,return_counts=True)
    rtimes,torontor = np.unique(torontorraw,return_counts=True)
    htimes,torontoh = np.unique(torontohraw,return_counts=True)
    ftimes,torontof = np.unique(torontofraw,return_counts=True)
    torontor = np.array(matchtimes(times,rtimes,torontor))
    torontoh = np.array(matchtimes(times,htimes,torontoh))
    torontof = np.array(matchtimes(times,ftimes,torontof))
    TOneighborhoods["casesTO"] = toronto
    TOneighborhoods["fatalTO"] = torontof
    TOneighborhoods["hosptTO"] = torontoh
    TOneighborhoods["recovTO"] = torontor
    for neighborhood in TOneighborhoods["units"]:
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["CASES"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["CASES"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["CASES"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["FATAL"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["FATAL"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["FATAL"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["HOSPITALIZED"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["HOSPITALIZED"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["HOSPITALIZED"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["RECOVERED"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["RECOVERED"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["RECOVERED"] = np.array(matchtimes(times,time,cases))

    popfile = open("neighbourhood-profiles-2016-csv.csv","r")
    popcsv = csv.reader(popfile,delimiter=',',quotechar='"',quoting=csv.QUOTE_ALL,skipinitialspace=True)
    n = 0
    for row in popcsv:
        if n==0:
            header = row[6:]
        elif n==3:
            poprow = row[6:]
            break
        n+=1
        
    neighborhoodpops = {}
    for n in range(len(header)):
        neighborhoodpops[header[n]] = float(poprow[n].replace(",",""))
    #neighborhoodpops.keys()

    if "" in TOneighborhoods["units"]:
        TOneighborhoods["units"].pop("")
        
    keys1 = sorted(neighborhoodpops.keys())
    keys2 = sorted(TOneighborhoods["units"].keys())
    #print(len(keys1),len(keys2))
    for n in range(len(keys1)):
        #print("%40s\t%s\t%s"%(keys1[n],str(keys1[n]==keys2[n]),keys2[n]))
        if keys1[n]!=keys2[n]:
            neighborhoodpops[keys2[n]] = neighborhoodpops[keys1[n]]
            neighborhoodpops.pop(keys1[n])
            
    for neighborhood in neighborhoodpops:
        TOneighborhoods["units"][neighborhood]["POP"] = neighborhoodpops[neighborhood]
            
    otimes = {}
    ontario = {}
    ontario_d = {}
    ontario_a = {}
    ontario_r = {}
    with open("ontario_cases.csv","r") as df:
        ontariocsv =df.read().split("\n")[1:]
        while ontariocsv[-1]=="":
            ontariocsv = ontariocsv[:-1]
    for line in ontariocsv:
        entry = line.split(',')
        if entry[1]=="":
            entry[1]="UNKNOWN"
        if entry[1][0]=='"':
            entry[1] = entry[1][1:]
        if entry[1][-1]=='"':
            entry[1] = entry[1][:-1]
        phu = entry[1]
        i0 = 0
        if entry[0][0]=='"':
            entry[0]=entry[0][1:]
        if entry[0][-1]=='"':
            entry[0]=entry[0][:-1]
        timestamp = entry[0].split('-')
        try:
            x = timestamp[1]
        except:
            timestamp = timestamp[0]
            timestamp = [timestamp[:4],timestamp[4:6],timestamp[6:]]
        if phu=='HALIBURTON' or phu=='KINGSTON':
            phu=entry[1]+','+entry[2]+','+entry[3]
            i0 = 2
        elif phu=='LEEDS':
            phu=entry[1]+','+entry[2]
            i0 = 1
        phu = phu.replace('"','')
        active = int(entry[3+i0])
        rec    = int(entry[4+i0])
        dead   = int(entry[5+i0])
        if phu not in otimes.keys():
            otimes[phu] = np.array([date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])),])
            ontario_a[phu] = np.array([active,])
            ontario_r[phu] = np.array([rec,])
            ontario_d[phu] = np.array([dead,])
        else:
            try:
                otimes[phu] = np.append(otimes[phu],date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])))
                ontario_a[phu] = np.append(ontario_a[phu],active)
                ontario_r[phu] = np.append(ontario_r[phu],rec)
                ontario_d[phu] = np.append(ontario_d[phu],dead)
            except:
                print(timestamp)
    for phu in otimes.keys():
        order = np.argsort(otimes[phu])
        otimes[phu] = otimes[phu][order]
        ontario_a[phu] = ontario_a[phu][order]
        ontario_r[phu] = ontario_r[phu][order]
        ontario_d[phu] = ontario_d[phu][order]
        ontario[phu] = (np.diff(np.append([0,],ontario_a[phu]))
                       +np.diff(np.append([0,],ontario_r[phu]))
                       +np.diff(np.append([0,],ontario_d[phu]))) 
                     #ontario_a[phu]+ontario_d[phu]+ontario_r[phu]

        
    TOneighborhoods["ProvincialTO"] = {"CASES":ontario["TORONTO"],
                                       "FATAL":ontario_d["TORONTO"],
                                       "ACTIVE":ontario_a["TORONTO"],
                                       "RECOVERED":ontario_r["TORONTO"]}
    
    time = ncd.createDimension("time",None)
    scalar = ncd.createDimension("scalar",1)
    
    ncd.set_auto_mask(False)
    
    nr = 100
    
    rdim = ncd.createDimension("Rt",nr)
    rrange = np.geomspace(0.1,10.0,num=nr)
    rbase = np.geomspace(0.01,10.0,num=1000)
    
    try:
        
        for country in sorted(countries):
            cdata = extract_country(dataset,country)
            if cdata["Total"][-1]>=25 and "Princess" not in country and "Olympics" not in country and "Zaandam" not in country:
                key = country
                if key=="US":
                    key="United States"
                ckey = str.title(key)
                countrygrp = ncd.createGroup(ckey)
                countrycases = ncd[ckey].createVariable("cases","i4",("time",),zlib=True)
                countrydeaths = ncd[ckey].createVariable("deaths","i4",("time",),zlib=True)
                countryRt = ncd[ckey].createVariable("Rt","f4",("time",),zlib=True)
                countrypopulation = ncd[ckey].createVariable("population","f4",("scalar",),zlib=True)
                ncd[ckey]["population"][:] = float(countrypops[country])
                ctotal = np.diff(np.append([0,],extract_country(dataset,country)["Total"])).astype(int)
                dtotal = np.diff(np.append([0,],extract_country(ddataset,country)["Total"])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal))
                ncd[ckey]["Rt"][:] = r
                ncd[ckey]["cases"][:] = ctotal
                ncd[ckey]["deaths"][:] = dtotal
                countrycases.set_auto_mask(False)
                countrydeaths.set_auto_mask(False)
                countryRt.set_auto_mask(False)
                countrypopulation.set_auto_mask(False)
                countrycases.units = "cases day-1"
                countrydeaths.units = "deaths day-1"
                countryRt.units = "secondary infections case-1"
                countrypopulation.units = "people"
                countrycases.standard_name = "daily_cases"
                countrydeaths.standard_name = "daily_deaths"
                countryRt.standard_name = "R_t"
                countrypopulation.standard_name = "population"
                countrycases.long_name = "New Cases per Day"
                countrydeaths.long_name = "New Deaths per Day"
                countryRt.long_name = "Effective Reproductive Number"
                countrypopulation.long_name = "Population"
                
                
        ncd.sync()
        
        for state in sorted(usa):
            if us_deaths[state][-1]>=20 and state!="Total" and "Princess" not in state\
                and "Virgin Islands" not in state and "Military" not in state\
                and "Recovered" not in state and "Prisons" not in state\
                and "Hospitals" not in state and state != "Total" and usa[state][-1]>150:
                    ckey = str.title(state)
                    stategrp = ncd["United States"].createGroup(ckey)
                    statecases = ncd["United States"][ckey].createVariable("cases","i2",("time",),zlib=True)
                    statedeaths = ncd["United States"][ckey].createVariable("deaths","i2",("time",),zlib=True)
                    stateRt = ncd["United States"][ckey].createVariable("Rt","f4",("time",),zlib=True)
                    statepopulation = ncd["United States"][ckey].createVariable("population","f4",("scalar",),zlib=True)
                    ncd["United States"][ckey]["population"][:] = float(statepops[state])
                    ctotal = np.diff(np.append([0,],usa[state])).astype(int)
                    dtotal = np.diff(np.append([0,],usa[state])).astype(int)
                    r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                    ncd["United States"][ckey]["Rt"][:] = r
                    ncd["United States"][ckey]["cases"][:] = ctotal
                    ncd["United States"][ckey]["deaths"][:] = dtotal
                    statecases.set_auto_mask(False)
                    statedeaths.set_auto_mask(False)
                    statepopulation.set_auto_mask(False)
                    statecases.units = "cases day-1"
                    statedeaths.units = "deaths day-1"
                    statepopulation.units = "people"
                    statecases.standard_name = "daily_cases"
                    statedeaths.standard_name = "daily_deaths"
                    statepopulation.standard_name = "population"
                    statecases.long_name = "new cases per day"
                    statedeaths.long_name = "new deaths per day"
                    statepopulation.long_name = "population"
        ncd.sync()
        
        for province in sorted(canada):
            if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total":
                ckey = str.title(province)
                ckey = ckey.replace("And","and")
                provincegrp = ncd["Canada"].createGroup(ckey)
                provincecases = ncd["Canada"][ckey].createVariable("cases","i2",("time",),zlib=True)
                provincedeaths = ncd["Canada"][ckey].createVariable("deaths","i2",("time",),zlib=True)
                provinceRt = ncd["Canada"][ckey].createVariable("Rt","f4",("time",),zlib=True)
                provincepopulation = ncd["Canada"][ckey].createVariable("population","f4",("scalar",),zlib=True)
                ncd["Canada"][ckey]["population"][:] = float(provincepops[province])
                ctotal = np.diff(np.append([0,],canada[province])).astype(int)
                dtotal = np.diff(np.append([0,],canada[province])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                ncd["Canada"][ckey]["Rt"][:] = r
                ncd["Canada"][ckey]["cases"][:] = ctotal
                ncd["Canada"][ckey]["deaths"][:] = dtotal
                provincecases.set_auto_mask(False)
                provincedeaths.set_auto_mask(False)
                provincepopulation.set_auto_mask(False)
                provincecases.units = "cases day-1"
                provincedeaths.units = "deaths day-1"
                provincepopulation.units = "people"
                provincecases.standard_name = "daily_cases"
                provincedeaths.standard_name = "daily_deaths"
                provincepopulation.standard_name = "population"
                provincecases.long_name = "new cases per day"
                provincedeaths.long_name = "new deaths per day"
                provincepopulation.long_name = "population"
        
        ncd.sync()
        
        for phu in sorted(ontario):
            if len(ontario[phu][:])>10:
                ckey = str.title(phu)
                phugrp = ncd["Canada"]["Ontario"].createGroup(ckey)
                phucases = ncd["Canada/Ontario"][ckey].createVariable("cases","i2",("time",),zlib=True)
                phudeaths = ncd["Canada/Ontario"][ckey].createVariable("deaths","i2",("time",),zlib=True)
                phuactive = ncd["Canada/Ontario"][ckey].createVariable("active","i2",("time",),zlib=True)
                phurecovered = ncd["Canada/Ontario"][ckey].createVariable("recovered","i2",("time",),zlib=True)
                phuRt = ncd["Canada/Ontario"][ckey].createVariable("Rt","f4",("time",),zlib=True)
                #phupopulation = ncd["Canada/Ontario"][ckey].createVariable("population","f4",("scalar",),zlib=True)
                #ncd["Canada/Ontario"][ckey]["population"][:] = float(phupops[phu])
                ctotal = ontario[phu].astype(int)
                dtotal = np.diff(np.append([0,],ontario_d[phu])).astype(int)
                atotal = ontario_a[phu].astype(int)
                rtotal = np.diff(np.append([0,],ontario_r[phu])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                ncd["Canada/Ontario"][ckey]["Rt"][:] = r
                ncd["Canada/Ontario"][ckey]["cases"][:] = ctotal
                ncd["Canada/Ontario"][ckey]["deaths"][:] = dtotal
                ncd["Canada/Ontario"][ckey]["active"][:] = atotal
                ncd["Canada/Ontario"][ckey]["recovered"][:] = rtotal
                phucases.set_auto_mask(False)
                phudeaths.set_auto_mask(False)
                phuactive.set_auto_mask(False)
                phurecovered.set_auto_mask(False)
                #phupopulation.set_auto_mask(False)
                phucases.units = "cases day-1"
                phudeaths.units = "deaths day-1"
                phuactive.units = "cases"
                phurecovered.units = "recoveries day-1"
                #phupopulation.units = "people"
                phucases.standard_name = "daily_cases"
                phudeaths.standard_name = "daily_deaths"
                phuactive.standard_name = "active_cases"
                phurecovered.standard_name = "daily_recoveries"
                #phupopulation.standard_name = "population"
                phucases.long_name = "new cases per day"
                phudeaths.long_name = "new deaths per day"
                phuactive.long_name = "active cases"
                phurecovered.long_name = "newly-recovered cases per day"
                #phupopulation.long_name = "population"
            
        TOpop = ncd["Canada/Ontario/Toronto"].createVariable("population","f4",("scalar",),zlib=True)
        TOpop[:] = 2.732e6
        TOpop.set_auto_mask(False)
        TOpop.units = "people"
        TOpop.standard_name="population"
        TOpop.long_name="population"
        
        TOhosp = ncd["Canada/Ontario/Toronto"].createVariable("hospitalized","i2",("time",),zlib=True)
        TOhosp[:] = TOneighborhoods["hosptTO"].astype(int)
        TOhosp.set_auto_mask(False)
        TOhosp.units = "hospitalizations day-1"
        TOhosp.standard_name = "daily_hospitalizations"
        TOhosp.long_name = "day's cases which ever required hospitalization"
        
        ncd.sync()
        
        for neighborhood in sorted(TOneighborhoods["units"]):
            ckey = neighborhood.replace("/","-")
            areagrp = ncd["Canada/Ontario/Toronto"].createGroup(ckey)
            areacases = ncd["Canada/Ontario/Toronto"][ckey].createVariable("cases","i2",("time",),zlib=True)
            areadeaths = ncd["Canada/Ontario/Toronto"][ckey].createVariable("deaths","i2",("time",),zlib=True)
            areahosp = ncd["Canada/Ontario/Toronto"][ckey].createVariable("hospitalized","i2",("time",),zlib=True)
            arearecovered = ncd["Canada/Ontario/Toronto"][ckey].createVariable("recovered","i2",("time",),zlib=True)
            areaRt = ncd["Canada/Ontario/Toronto"][ckey].createVariable("Rt","f4",("time",),zlib=True)
            areapopulation = ncd["Canada/Ontario/Toronto"][ckey].createVariable("population","f4",("scalar",),zlib=True)
            ncd["Canada/Ontario/Toronto"][ckey]["population"][:] = float(TOneighborhoods["units"][neighborhood]["POP"])
            ctotal = TOneighborhoods["units"][neighborhood]["CASES"].astype(int)
            dtotal = TOneighborhoods["units"][neighborhood]["FATAL"].astype(int)
            htotal = TOneighborhoods["units"][neighborhood]["HOSPITALIZED"].astype(int)
            rtotal = TOneighborhoods["units"][neighborhood]["RECOVERED"].astype(int)
            r,lp,ll = Rt(day5avg(ctotal.astype(float)))
            ncd["Canada/Ontario/Toronto"][ckey]["Rt"][:] = r
            ncd["Canada/Ontario/Toronto"][ckey]["cases"][:] = ctotal
            ncd["Canada/Ontario/Toronto"][ckey]["deaths"][:] = dtotal
            ncd["Canada/Ontario/Toronto"][ckey]["hospitalized"][:] = htotal
            ncd["Canada/Ontario/Toronto"][ckey]["recovered"][:] = rtotal
            areacases.set_auto_mask(False)
            areadeaths.set_auto_mask(False)
            areahosp.set_auto_mask(False)
            arearecovered.set_auto_mask(False)
            areapopulation.set_auto_mask(False)
            areacases.units = "cases day-1"
            areadeaths.units = "deaths day-1"
            areahosp.units = "hospitalizations day-1"
            arearecovered.units = "recoveries day-1"
            areapopulation.units = "people"
            areacases.standard_name = "daily_cases"
            areadeaths.standard_name = "daily_deaths"
            areahosp.standard_name = "daily_hospitalizations"
            arearecovered.standard_name = "daily_recoveries"
            areapopulation.standard_name = "population"
            areacases.long_name = "new cases per day"
            areadeaths.long_name = "new deaths per day"
            areahosp.long_name = "day's cases which ever required hospitalization"
            arearecovered.long_name = "day's cases which have recovered"
            areapopulation.long_name = "population"
            
        ncd.sync()
        
        for state in sorted(usa):
            if us_deaths[state][-1]>=20 and state!="Total" and "Princess" not in state\
                and "Virgin Islands" not in state and "Military" not in state\
                and "Recovered" not in state and "Prisons" not in state and state!="Guam"\
                and "Hospitals" not in state and state != "Total" and usa[state][-1]>150:
                
                counties = get_counties(usacsv,state=state)
                for county in counties:
                    ckey = county.replace("/","-")
                    countygrp = ncd["United States/%s"%str.title(state)].createGroup(ckey)
                    countycases = ncd["United States/%s"%str.title(state)][ckey].createVariable("cases","i2",("time",),zlib=True)
                    countyRt = ncd["United States/%s"%str.title(state)][ckey].createVariable("Rt","f4",("time",),zlib=True)
                    countypopulation = ncd["United States/%s"%str.title(state)][ckey].createVariable("population","f4",("scalar",),zlib=True)
                    ncd["United States/%s"%str.title(state)][ckey]["population"][:] = float(get_countypop(county,state))
                    ctotal = np.diff(np.append([0,],extract_county(usacsv,county,state=state))).astype(int)
                    r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                    ncd["United States/%s"%str.title(state)][ckey]["Rt"][:] = r
                    ncd["United States/%s"%str.title(state)][ckey]["cases"][:] = ctotal
                    countycases.set_auto_mask(False)
                    countypopulation.set_auto_mask(False)
                    countycases.units = "cases day-1"
                    countypopulation.units = "people"
                    countycases.standard_name = "daily_cases"
                    countypopulation.standard_name = "population"
                    countycases.long_name = "New Cases per Day"
                    countypopulation.long_name = "Population"
                    
                ncd.sync()
    except:
        ncd.close()
        raise
    ncd.close()
       
def hdf5_slim():
    import h5py as h5
    
    countrysetf = "countrypopulations.csv"
    with open(countrysetf,"r") as df:
        countryset = df.read().split('\n')[1:]
    if countryset[-1] == "":
        countryset = countryset[:-1]
    countrypops = {}
    for line in countryset:
        linedata = line.split(',')
        name = linedata[0]
        popx = float(linedata[4])
        countrypops[name] = popx
    
    canadasetf = "provincepopulations.csv"
    with open(canadasetf,"r") as df:
        canadaset = df.read().split('\n')[2:]
    if canadaset[-1] == "":
        canadaset = canadaset[:-1]
    provincepops = {}
    for line in canadaset:
        linedata = line.split(',')
        name = linedata[1]
        popx = float(linedata[-3])
        provincepops[name] = popx
        
    statesetf = "statepopulations.csv"
    with open(statesetf,"r") as df:
        stateset = df.read().split('\n')[2:]
    if stateset[-1] == "":
        stateset = stateset[:-1]
    statepops = {}
    for line in stateset:
        linedata = line.split(',')
        name = linedata[2]
        popx = float(linedata[3])
        statepops[name] = popx
        
    #_log("/home/adivp416/public_html/covid19/reportlog.txt","Static CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
        
    globalconf = "github/time_series_covid19_confirmed_global.csv"

    with open(globalconf,"r") as df:
        dataset = df.read().split('\n')
    header = dataset[0].split(',')
    dataset = dataset[1:]
    if dataset[-1]=='':
        dataset = dataset[:-1]
    latestglobal = header[-1]
        
    countries = getcountries(dataset)
    

    usacsv = []
    with open("github/time_series_covid19_confirmed_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usacsv.append(row)
            
    usa = extract_usa(usacsv)

    canada = extract_country(dataset,"Canada")

    ddatasetf = "github/time_series_covid19_deaths_global.csv"

    with open(ddatasetf,"r") as df:
        ddataset = df.read().split('\n')
    header = ddataset[0].split(',')
    ddataset = ddataset[1:]
    if ddataset[-1]=='':
        ddataset = ddataset[:-1]
        
    usadcsv = []
    with open("github/time_series_covid19_deaths_US.csv") as csvfile:
        creader = csv.reader(csvfile,delimiter=',',quotechar='"')
        for row in creader:
            usadcsv.append(row)
            
    #_log("/home/adivp416/public_html/covid19/reportlog.txt","Dynamic CSVs loaded. \t%s"%systime.asctime(systime.localtime()))
    
    ca_deaths = extract_country(ddataset,"Canada")
    us_deaths = extract_usa(usadcsv)

    timestamps = []
    rtimes = []
    ftimes = []
    htimes = []
    TOneighborhoods = {"units":{}}
    with open("toronto_cases.csv","r") as df:
        torontocsv = df.read().split('\n')[1:]
        while torontocsv[-1]=="":
            torontocsv = torontocsv[:-1]
        torontoraw = np.zeros(len(torontocsv))
        for line in range(len(torontoraw)):
            timestamp = torontocsv[line].split(',')[9].split('-')
            timestamps.append(date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])))
            entry = torontocsv[line].split(',')
            if entry[11]=="ACTIVE":
                pass
            elif entry[11]=="FATAL":
                ftimes.append(timestamps[-1])
            else:
                rtimes.append(timestamps[-1])
            if entry[15]=="Yes":
                htimes.append(timestamps[-1])
                
            if entry[4] not in TOneighborhoods["units"]:
                TOneighborhoods["units"][entry[4]] = {"CASES":[],"FATAL":[],"HOSPITALIZED":[],"RECOVERED":[]}
            TOneighborhoods["units"][entry[4]]["CASES"].append(timestamps[-1])
            if entry[11]=="ACTIVE":
                pass
            elif entry[11]=="FATAL":
                TOneighborhoods["units"][entry[4]]["FATAL"].append(timestamps[-1])
            else:
                TOneighborhoods["units"][entry[4]]["RECOVERED"].append(timestamps[-1])
            if entry[15]=="Yes":
                TOneighborhoods["units"][entry[4]]["HOSPITALIZED"].append(timestamps[-1])
            
    #for neighborhood in TOneighborhoods["units"]:
    #    TOneighborhoods["units"][neighborhood]["CASES"] = np.array(TOneighborhoods["units"][neighborhood]["CASES"])
    #    TOneighborhoods["units"][neighborhood]["FATAL"] = np.array(TOneighborhoods["units"][neighborhood]["FATAL"])
    #    TOneighborhoods["units"][neighborhood]["HOSPITALIZED"] = np.array(TOneighborhoods["units"][neighborhood]["HOSPITALIZED"])
    #    TOneighborhoods["units"][neighborhood]["RECOVERED"] = np.array(TOneighborhoods["units"][neighborhood]["RECOVERED"])
            
    #All cases
    origin = np.array(timestamps).min()
    for line in range(len(timestamps)):
        day = ((timestamps[line]-origin).days)
        torontoraw[line] = int(day)
    #Recovered cases
    torontorraw = np.zeros(len(rtimes))
    for line in range(len(rtimes)):
        day = ((rtimes[line]-origin).days)
        torontorraw[line] = int(day)
    #Hospitalized cases
    torontohraw = np.zeros(len(htimes))
    for line in range(len(htimes)):
        day = ((htimes[line]-origin).days)
        torontohraw[line] = int(day)
    #Fatal cases
    torontofraw = np.zeros(len(ftimes))
    for line in range(len(ftimes)):
        day = ((ftimes[line]-origin).days)
        torontofraw[line] = int(day)
    
    timestamps = np.array(timestamps)
    times,toronto = np.unique(torontoraw,return_counts=True)
    rtimes,torontor = np.unique(torontorraw,return_counts=True)
    htimes,torontoh = np.unique(torontohraw,return_counts=True)
    ftimes,torontof = np.unique(torontofraw,return_counts=True)
    torontor = np.array(matchtimes(times,rtimes,torontor))
    torontoh = np.array(matchtimes(times,htimes,torontoh))
    torontof = np.array(matchtimes(times,ftimes,torontof))
    TOneighborhoods["casesTO"] = toronto
    TOneighborhoods["fatalTO"] = torontof
    TOneighborhoods["hosptTO"] = torontoh
    TOneighborhoods["recovTO"] = torontor
    for neighborhood in TOneighborhoods["units"]:
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["CASES"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["CASES"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["CASES"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["FATAL"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["FATAL"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["FATAL"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["HOSPITALIZED"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["HOSPITALIZED"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["HOSPITALIZED"] = np.array(matchtimes(times,time,cases))
        raw  = np.zeros(len(TOneighborhoods["units"][neighborhood]["RECOVERED"]))
        for line in range(len(raw)):
            day = (TOneighborhoods["units"][neighborhood]["RECOVERED"][line]-origin).days
            raw[line] = int(day)
        time,cases = np.unique(raw,return_counts=True)
        TOneighborhoods["units"][neighborhood]["RECOVERED"] = np.array(matchtimes(times,time,cases))

    popfile = open("neighbourhood-profiles-2016-csv.csv","r")
    popcsv = csv.reader(popfile,delimiter=',',quotechar='"',quoting=csv.QUOTE_ALL,skipinitialspace=True)
    n = 0
    for row in popcsv:
        if n==0:
            header = row[6:]
        elif n==3:
            poprow = row[6:]
            break
        n+=1
        
    neighborhoodpops = {}
    for n in range(len(header)):
        neighborhoodpops[header[n]] = float(poprow[n].replace(",",""))
    #neighborhoodpops.keys()

    if "" in TOneighborhoods["units"]:
        TOneighborhoods["units"].pop("")
        
    keys1 = sorted(neighborhoodpops.keys())
    keys2 = sorted(TOneighborhoods["units"].keys())
    #print(len(keys1),len(keys2))
    for n in range(len(keys1)):
        #print("%40s\t%s\t%s"%(keys1[n],str(keys1[n]==keys2[n]),keys2[n]))
        if keys1[n]!=keys2[n]:
            neighborhoodpops[keys2[n]] = neighborhoodpops[keys1[n]]
            neighborhoodpops.pop(keys1[n])
            
    for neighborhood in neighborhoodpops:
        TOneighborhoods["units"][neighborhood]["POP"] = neighborhoodpops[neighborhood]
            
    otimes = {}
    ontario = {}
    ontario_d = {}
    ontario_a = {}
    ontario_r = {}
    with open("ontario_cases.csv","r") as df:
        ontariocsv =df.read().split("\n")[1:]
        while ontariocsv[-1]=="":
            ontariocsv = ontariocsv[:-1]
    for line in ontariocsv:
        entry = line.split(',')
        if entry[1]=="":
            entry[1]="UNKNOWN"
        if entry[1][0]=='"':
            entry[1] = entry[1][1:]
        if entry[1][-1]=='"':
            entry[1] = entry[1][:-1]
        phu = entry[1]
        i0 = 0
        if entry[0][0]=='"':
            entry[0]=entry[0][1:]
        if entry[0][-1]=='"':
            entry[0]=entry[0][:-1]
        timestamp = entry[0].split('-')
        try:
            x = timestamp[1]
        except:
            timestamp = timestamp[0]
            timestamp = [timestamp[:4],timestamp[4:6],timestamp[6:]]
        if phu=='HALIBURTON' or phu=='KINGSTON':
            phu=entry[1]+','+entry[2]+','+entry[3]
            i0 = 2
        elif phu=='LEEDS':
            phu=entry[1]+','+entry[2]
            i0 = 1
        phu = phu.replace('"','')
        active = int(entry[3+i0])
        rec    = int(entry[4+i0])
        dead   = int(entry[5+i0])
        if phu not in otimes.keys():
            otimes[phu] = np.array([date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])),])
            ontario_a[phu] = np.array([active,])
            ontario_r[phu] = np.array([rec,])
            ontario_d[phu] = np.array([dead,])
        else:
            try:
                otimes[phu] = np.append(otimes[phu],date(int(timestamp[0]),int(timestamp[1]),int(timestamp[2])))
                ontario_a[phu] = np.append(ontario_a[phu],active)
                ontario_r[phu] = np.append(ontario_r[phu],rec)
                ontario_d[phu] = np.append(ontario_d[phu],dead)
            except:
                print(timestamp)
    for phu in otimes.keys():
        order = np.argsort(otimes[phu])
        otimes[phu] = otimes[phu][order]
        ontario_a[phu] = ontario_a[phu][order]
        ontario_r[phu] = ontario_r[phu][order]
        ontario_d[phu] = ontario_d[phu][order]
        ontario[phu] = (np.diff(np.append([0,],ontario_a[phu]))
                       +np.diff(np.append([0,],ontario_r[phu]))
                       +np.diff(np.append([0,],ontario_d[phu]))) 
                     #ontario_a[phu]+ontario_d[phu]+ontario_r[phu]

    latestON = otimes["TORONTO"].max() #datetime.date
    latestTO = timestamps.max()        #datetime.date
    latestusa = usacsv[0][-1]
    usatime = latestusa.split("/")
    latestusa = date(2000+int(usatime[2]),int(usatime[0]),int(usatime[1]))
    globaltime = latestglobal.split("/")
    latestglobal = date(2000+int(globaltime[2]),int(globaltime[0]),int(globaltime[1]))
    latestON = ' '.join([x for i,x in enumerate(latestON.ctime().split()) if i!=3])
    latestTO = ' '.join([x for i,x in enumerate(latestTO.ctime().split()) if i!=3])
    latestusa = ' '.join([x for i,x in enumerate(latestusa.ctime().split()) if i!=3])
    latestglobal = ' '.join([x for i,x in enumerate(latestglobal.ctime().split()) if i!=3])
        
    TOneighborhoods["ProvincialTO"] = {"CASES":ontario["TORONTO"],
                                       "FATAL":ontario_d["TORONTO"],
                                       "ACTIVE":ontario_a["TORONTO"],
                                       "RECOVERED":ontario_r["TORONTO"]}
    
    hdf = h5.File("adivparadise_covid19data_slim.hdf5","w")
    
    nr = 100
    
    rrange = np.geomspace(0.1,10.0,num=nr)
    hdf.create_dataset("/rtrange",data=rrange.astype("float32"),compression='gzip',
                       compression_opts=9,shuffle=True,fletcher32=True)
    hdf["rtrange"].attrs['long_name'] = "Possible R$_t$ Values"
    rbase = np.geomspace(0.01,10.0,num=1000)
    
    
    try:
        for country in sorted(countries):
            cdata = extract_country(dataset,country)
            if cdata["Total"][-1]>=25 and "Princess" not in country and "Olympics" not in country and "Zaandam" not in country:
                key = country
                if key=="US":
                    key="United States"
                ckey = str.title(key)
                ctotal = np.diff(np.append([0,],extract_country(dataset,country)["Total"])).astype(int)
                dtotal = np.diff(np.append([0,],extract_country(ddataset,country)["Total"])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal))
                p = np.exp(lp)
                l = np.exp(ll)
                pd = np.zeros((p.shape[0],nr))
                ld = np.zeros((l.shape[0],nr))
                for t in range(p.shape[0]):
                    pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                    ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                    
                countrycases = hdf.create_dataset("/"+ckey+"/cases",compression='gzip',compression_opts=9,
                                                  shuffle=True,fletcher32=True,data=ctotal)
                countrydeaths = hdf.create_dataset("/"+ckey+"/deaths",compression='gzip',compression_opts=9,
                                                  shuffle=True,fletcher32=True,data=dtotal)
                countryRt = hdf.create_dataset("/"+ckey+"/Rt",compression='gzip',compression_opts=9,
                                                  shuffle=True,fletcher32=True,data=r.astype("float32"))
                countrypopulation = hdf.create_dataset("/"+ckey+"/population",data=float(countrypops[country]))
                latest = hdf.create_dataset("/"+ckey+"/latestdate",data=latestglobal)

                countrycases.attrs["units"] = "cases day-1"
                countrydeaths.attrs["units"] = "deaths day-1"
                countryRt.attrs["units"] = "secondary infections case-1"
                countrypopulation.attrs["units"] = "people"
                countrycases.attrs["standard_name"] = "daily_cases"
                countrydeaths.attrs["standard_name"] = "daily_deaths"
                countryRt.attrs["standard_name"] = "R_t"
                countrypopulation.attrs["standard_name"] = "population"
                countrycases.attrs["long_name"] = "New Cases per Day"
                countrydeaths.attrs["long_name"] = "New Deaths per Day"
                countryRt.attrs["long_name"] = "Effective Reproductive Number"
                countrypopulation.attrs["long_name"] = "Population"
                
           
        for state in sorted(usa):
            if us_deaths[state][-1]>=20 and state!="Total" and "Princess" not in state\
                and "Virgin Islands" not in state and "Military" not in state\
                and "Recovered" not in state and "Prisons" not in state\
                and "Hospitals" not in state and state != "Total" and usa[state][-1]>150:
                    ckey = str.title(state)
                    ctotal = np.diff(np.append([0,],usa[state])).astype(int)
                    dtotal = np.diff(np.append([0,],usa[state])).astype(int)
                    r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                    p = np.exp(lp)
                    l = np.exp(ll)
                    pd = np.zeros((p.shape[0],nr))
                    ld = np.zeros((l.shape[0],nr))
                    for t in range(p.shape[0]):
                        pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                        ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                        
                    statecases  = hdf.create_dataset("/United States/"+ckey+"/cases",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ctotal.astype(np.short))
                    statedeaths = hdf.create_dataset("/United States/"+ckey+"/deaths",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=dtotal.astype(np.short))
                    stateRt     = hdf.create_dataset("/United States/"+ckey+"/Rt",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=r.astype("float32"))
                    statepopulation = hdf.create_dataset("/United States/"+ckey+"/population",data=float(statepops[state]))
                    latest = hdf.create_dataset("/United States/"+ckey+"/latestdate",data=latestusa)
  
                    statecases.attrs["units"] = "cases day-1"
                    statedeaths.attrs["units"] = "deaths day-1"
                    statepopulation.attrs["units"] = "people"
                    statecases.attrs["standard_name"] = "daily_cases"
                    statedeaths.attrs["standard_name"] = "daily_deaths"
                    statepopulation.attrs["standard_name"] = "population"
                    statecases.attrs["long_name"] = "new cases per day"
                    statedeaths.attrs["long_name"] = "new deaths per day"
                    statepopulation.attrs["long_name"] = "population"
   
           
        for province in sorted(canada):
            if "Princess" not in province and province!="Recovered" and province!="Repatriated Travellers" and province != "Total":
                ckey = str.title(province)
                ckey = ckey.replace("And","and")
                ctotal = np.diff(np.append([0,],canada[province])).astype(int)
                dtotal = np.diff(np.append([0,],canada[province])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                p = np.exp(lp)
                l = np.exp(ll)
                pd = np.zeros((p.shape[0],nr))
                ld = np.zeros((l.shape[0],nr))
                for t in range(p.shape[0]):
                    pd[t,:] = np.interp(rrange[:],rbase,p[t,:])
                    ld[t,:] = np.interp(rrange[:],rbase,l[t,:])
                    
                provincecases  = hdf.create_dataset("/Canada/"+ckey+"/cases",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ctotal.astype(np.short))
                provincedeaths = hdf.create_dataset("/Canada/"+ckey+"/deaths",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=dtotal.astype(np.short))
                provinceRt     = hdf.create_dataset("/Canada/"+ckey+"/Rt",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=r.astype("float32"))
                provincepopulation = hdf.create_dataset("/Canada/"+ckey+"/population",data=float(provincepops[province]))
                latest = hdf.create_dataset("/Canada/"+ckey+"/latestdate",data=latestglobal)

                provincecases.attrs["units"] = "cases day-1"
                provincedeaths.attrs["units"] = "deaths day-1"
                provincepopulation.attrs["units"] = "people"
                provincecases.attrs["standard_name"] = "daily_cases"
                provincedeaths.attrs["standard_name"] = "daily_deaths"
                provincepopulation.attrs["standard_name"] = "population"
                provincecases.attrs["long_name"] = "new cases per day"
                provincedeaths.attrs["long_name"] = "new deaths per day"
                provincepopulation.attrs["long_name"] = "population"

        
        
        for phu in sorted(ontario):
            if len(ontario[phu][:])>10:
                ckey = str.title(phu)
                ctotal = ontario[phu].astype(int)
                dtotal = np.diff(np.append([0,],ontario_d[phu])).astype(int)
                atotal = ontario_a[phu].astype(int)
                rtotal = np.diff(np.append([0,],ontario_r[phu])).astype(int)
                r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                phucases = hdf.create_dataset("/Canada/Ontario/"+ckey+"/cases",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ctotal.astype(np.short))
                phudeaths = hdf.create_dataset("/Canada/Ontario/"+ckey+"/deaths",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=dtotal.astype(np.short))
                phuactive = hdf.create_dataset("/Canada/Ontario/"+ckey+"/active",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=atotal.astype(np.short))
                phurecovered = hdf.create_dataset("/Canada/Ontario/"+ckey+"/recovered",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=rtotal.astype(np.short))
                phuRt = hdf.create_dataset("/Canada/Ontario/"+ckey+"/Rt",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=r.astype('float32'))
                latest = hdf.create_dataset("/Canada/Ontario/"+ckey+"/latestdate",data=latestON)
                #phupopulation = ncd["Canada/Ontario"][ckey].createVariable("population","f4",("scalar",),zlib=True)
                #ncd["Canada/Ontario"][ckey]["population"][:] = float(phupops[phu])
                #phupopulation.set_auto_mask(False)
                phucases.attrs["units"] = "cases day-1"
                phudeaths.attrs["units"] = "deaths day-1"
                phuactive.attrs["units"] = "cases"
                phurecovered.attrs["units"] = "recoveries day-1"
                #phupopulation.units = "people"
                phucases.attrs["standard_name"] = "daily_cases"
                phudeaths.attrs["standard_name"] = "daily_deaths"
                phuactive.attrs["standard_name"] = "active_cases"
                phurecovered.attrs["standard_name"] = "daily_recoveries"
                #phupopulation.standard_name = "population"
                phucases.attrs["long_name"] = "new cases per day"
                phudeaths.attrs["long_name"] = "new deaths per day"
                phuactive.attrs["long_name"] = "active cases"
                phurecovered.attrs["long_name"] = "newly-recovered cases per day"
                #phupopulation.long_name = "population"
            
        TOpop = hdf.create_dataset("/Canada/Ontario/Toronto/population",data=2.732e6)

        TOpop.attrs["units"] = "people"
        TOpop.attrs["standard_name"]="population"
        TOpop.attrs["long_name"]="population"
        
        TOhosp = hdf.create_dataset("/Canada/Ontario/Toronto/hospitalized",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=TOneighborhoods["hosptTO"].astype(np.short))

        TOhosp.attrs["units"] = "hospitalizations day-1"
        TOhosp.attrs["standard_name"] = "daily_hospitalizations"
        TOhosp.attrs["long_name"] = "day's cases which ever required hospitalization"
        
        
        for neighborhood in sorted(TOneighborhoods["units"]):
            ckey = neighborhood.replace("/","-")
            ctotal = TOneighborhoods["units"][neighborhood]["CASES"].astype(int)
            dtotal = TOneighborhoods["units"][neighborhood]["FATAL"].astype(int)
            htotal = TOneighborhoods["units"][neighborhood]["HOSPITALIZED"].astype(int)
            rtotal = TOneighborhoods["units"][neighborhood]["RECOVERED"].astype(int)
            r,lp,ll = Rt(day5avg(ctotal.astype(float)))

            areacases = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/cases",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=ctotal.astype(np.short))
            areadeaths = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/deaths",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=dtotal.astype(np.short))
            areahosp = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/hospitalized",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=htotal.astype(np.short))
            arearecovered = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/recovered",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=rtotal.astype(np.short))
            areaRt = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/Rt",compression='gzip',
                                                     compression_opts=9,shuffle=True,fletcher32=True,
                                                     data=r.astype("float32"))
            areapopulation = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/population",
                                                data=float(TOneighborhoods["units"][neighborhood]["POP"]))
            latest = hdf.create_dataset("/Canada/Ontario/Toronto/"+ckey+"/latestdate",data=latestTO)

            areacases.attrs["units"] = "cases day-1"
            areadeaths.attrs["units"] = "deaths day-1"
            areahosp.attrs["units"] = "hospitalizations day-1"
            arearecovered.attrs["units"] = "recoveries day-1"
            areapopulation.attrs["units"] = "people"
            areacases.attrs["standard_name"] = "daily_cases"
            areadeaths.attrs["standard_name"] = "daily_deaths"
            areahosp.attrs["standard_name"] = "daily_hospitalizations"
            arearecovered.attrs["standard_name"] = "daily_recoveries"
            areapopulation.attrs["standard_name"] = "population"
            areacases.attrs["long_name"] = "new cases per day"
            areadeaths.attrs["long_name"] = "new deaths per day"
            areahosp.attrs["long_name"] = "day's cases which ever required hospitalization"
            arearecovered.attrs["long_name"] = "day's cases which have recovered"
            areapopulation.attrs["long_name"] = "population"
                  
        for state in sorted(usa):
            if us_deaths[state][-1]>=20 and state!="Total" and "Princess" not in state\
                and "Virgin Islands" not in state and "Military" not in state\
                and "Recovered" not in state and "Prisons" not in state and state!="Guam"\
                and "Hospitals" not in state and state != "Total" and usa[state][-1]>150:
                
                counties = get_counties(usacsv,state=state)
                for county in counties:
                    ckey = county.replace("/","-")
                    ctotal = np.diff(np.append([0,],extract_county(usacsv,county,state=state))).astype(int)
                    r,lp,ll = Rt(day5avg(ctotal.astype(float)))
                        
                    countycases = hdf.create_dataset("/United States/%s"%str.title(state)+"/"+ckey+"/cases",
                                                     compression='gzip',compression_opts=9,shuffle=True,
                                                     fletcher32=True,data=ctotal.astype(np.short))
                    countyRt = hdf.create_dataset("/United States/%s"%str.title(state)+"/"+ckey+"/Rt",
                                                     compression='gzip',compression_opts=9,shuffle=True,
                                                     fletcher32=True,data=r.astype('float'))
                    countypopulation = hdf.create_dataset("/United States/%s"%str.title(state)+"/"+ckey+"/population",
                                                     data=float(get_countypop(county,state)))
                    latest = hdf.create_dataset("/United States/%s"%str.title(state)+"/"+ckey+"/latestdate",
                                                data = latestusa)

                    countycases.attrs["units"] = "cases day-1"
                    countypopulation.attrs["units"] = "people"
                    countycases.attrs["standard_name"] = "daily_cases"
                    countypopulation.attrs["standard_name"] = "population"
                    countycases.attrs["long_name"] = "New Cases per Day"
                    countypopulation.attrs["long_name"] = "Population"
        
        import matplotlib.pyplot as plt
        cases = hdf["United Kingdom/cases"][:]
        population = hdf["United Kingdom/population"][()]
        latest_date = hdf["United Kingdom/latestdate"][()]
        
        cases_per_100k = cases/population * 1.0e5 #Multiply by 100k to get cases per 100k
        
        cases_per_100k_7day = np.convolve(cases_per_100k,np.ones(7),'valid')/7.0
        
        days_before = np.arange(len(cases_per_100k_7day)) - len(cases_per_100k_7day) #max value is now 0
        
        #Create the plot
        fig,axes = plt.subplots(figsize=(14,8))
        axes.plot(days_before, cases_per_100k_7day)
        axes.set_xlabel("Days before "+latest_date)
        axes.set_ylabel("7-day Average of Daily Cases per 100k")
        axes.set_title("Average Daily cases for the UK as of "+latest_date)
        plt.savefig("UK_example.png",bbox_inches='tight',facecolor='white')
        
    except:
        hdf.close()
        raise
    
    hdf.close()
    
       
        
    
    
    

if __name__=="__main__":
    import os 
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    import numpy as np
    from scipy.special import factorial,loggamma
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv
    import time as systime
    from datetime import date, timedelta
    import warnings,traceback
    import sys
    warnings.filterwarnings('ignore')
    
    if "report" in sys.argv[:]:
        report()
    if "counties" in sys.argv[:]:
        processcounties(sys.argv[2]) #call pattern python covid19report.py counties Minnesota
    if "makeshell" in sys.argv[:]:
        makeshell()
    if "datasets" in sys.argv[:]:
        #netcdf_slim()
        #netcdf()
        hdf5_slim()
        hdf5()
        
