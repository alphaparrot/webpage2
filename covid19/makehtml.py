import os, time

imgplate = ('\t\t\t\t<div class="image full">                               \n'+
            '\t\t\t\t    <img src="%s" alt="%s">                            \n'+
            '\t\t\t\t    <div class="description">                          \n'+
            '\t\t\t\t    <p class="description content"> %s <br/>           \n'+
            '\t\t\t\t    <a href="%s" target="_blank">PNG</a> | <a href="%s" target="_blank">PDF</a></p> \n'+
            '\t\t\t\t    </div>                                \n '+
            '\t\t\t\t</div> ')

img2plate = ('\t\t\t\t<div class="image full">                              \n'+
            '\t\t\t\t    <div class="flex-container">                       \n'+
            '\t\t\t\t    <div class="column"><img src="%s" alt="%s"></div>  \n'+
            '\t\t\t\t    <div class="column"><img src="%s" alt="%s"></div>  \n'+
            '\t\t\t\t    </div>                                             \n'+
            '\t\t\t\t    <div class="description">                          \n'+
            '\t\t\t\t    <p class="description content"> %s <br/>           \n'+
            '\t\t\t\t    Left: <a href="%s" target="_blank">PNG</a> | <a href="%s" target="_blank">PDF</a><br/> \n'+
            '\t\t\t\t    Right: <a href="%s" target="_blank">PNG</a> | <a href="%s" target="_blank">PDF</a></p> \n'+
            '\t\t\t\t    </div>                                             \n'+
            '\t\t\t\t</div>  ')
#imgplate%(filepath,alttext,description,pdflink,pnglink)
#imgplate%(filepath1,alttext1,filepath2,alttext2,description,pdflink1,pnglink1,pdflink2,pnglink2)

par = "<p>%s</p>"

section = '<p style="font-size:25px"><a name="%s">%s</a></p>'
#section%(name,sectiontitle)

def makeneighborhood(neighborhood,pathdir):
    with open("template.html","r") as templatef:
        template = templatef.read().split('\n')
    filename = neighborhood.replace("/","_").replace(" ","_").replace("-","_")+".html"
    title = "\t\t<title>%s | COVID-19 Dashboard</title>"%neighborhood
    navlink = '\t\t\t\t\t\t\t<li class="active"><a href="%s">%s, Toronto</a></li>'%(filename,neighborhood)
    header = '\t\t\t\t\t<h2><a name="current">%s COVID-19 Data</a></h2>'%neighborhood
    
    body = ("<header><h2>%s Plots</h2></header>    \n"%neighborhood+
            imgplate%("%s_rawcases.png"%pathdir,"%s cases per day"%neighborhood,
                      "New COVID-19 cases per day in %s, Toronto. These data "%neighborhood+
                      "have not had a rolling average applied.","%s_rawcases.png"%pathdir,
                      "%s_rawcases.pdf"%pathdir)+"\n"+
            "<br> \n"+
            img2plate%("%s_avgcases.png"%pathdir,
                      "%s average cases per day, linear scale"%neighborhood,
                      "%s_avgcases_log.png"%pathdir,
                      "%s_average cases per day, logarithmic scale"%neighborhood,
                      "7-day average of new COVID-19 cases per day in %s, Toronto."%neighborhood+
                      "These data have had a rolling average applied. The first plot gives the data on a linear vertical scale, while the second gives the data on a logarithmic vertical scale, which better shows different exponentials.",
                      "%s_avgcases.png"%pathdir,"%s_avgcases.pdf"%pathdir,
                      "%s_avgcases_log.png"%pathdir,"%s_avgcases_log.pdf"%pathdir)+"\n"+
            "<br> \n"+
            img2plate%("%s_breakdown.png"%pathdir,
                      "%s case breakdown, linear scale"%neighborhood,
                      "%s_breakdown_log.png"%pathdir,
                      "%s_case breakdown, logarithmic scale"%neighborhood,
                      "Breakdown of daily COVID-19 cases per day in %s, Toronto by status--"%neighborhood+
                      "active, recovered, ever hospitalized, or dead. Note that 'hospitalized' is a status that applies to any of the other 3 categories. These data have not had a rolling average applied. The first plot gives the data on a linear vertical scale, while the second gives the data on a logarithmic vertical scale, which better shows different exponentials.",
                      "%s_breakdown.png"%pathdir,"%s_breakdown.pdf"%pathdir,
                      "%s_breakdown_log.png"%pathdir,"%s_breakdown_log.pdf"%pathdir)+"\n"+
            "<br> \n"+
            imgplate%("%s_Rt.png"%pathdir,"Historical %s effective reproductive number"%neighborhood,
                      "Raw and 2-week average of %s and Toronto's effective reproductive numbers. This is the average number of people a sick person will infect. If this is increasing, then transmission is increasing, even if cases are still declining. If this is above 1, then cases are increasing. If it is decreasing, then transmission is declining, even if cases are still rising. Note that due to the existence of super-spreaders, this metric is not the same as the number of people the average sick person will infect (i.e. a person selected at random from the cohort of infected people will typically infect fewer people than would be implied by R<sub>t</sub>, but a small fraction will infect many more."%neighborhood,"%s_Rt.png"%pathdir,
                      "%s_Rt.pdf"%pathdir)+"\n"+
            "<br> \n"+
            img2plate%("%s_relcases.png"%pathdir,
                      "%s average cases per day per 100k, linear scale"%neighborhood,
                      "%s_relcases_log.png"%pathdir,
                      "%s_average cases per day per 100k, logarithmic scale"%neighborhood,
                      "7-day average of new COVID-19 cases per day per 100k in %s, Toronto."%neighborhood+
                      "These data have had a rolling average applied. The first plot gives the data on a linear vertical scale, while the second gives the data on a logarithmic vertical scale, which better shows different exponentials.",
                      "%s_relcases.png"%pathdir,"%s_relcases.pdf"%pathdir,
                      "%s_relcases_log.png"%pathdir,"%s_relcases_log.pdf"%pathdir)+"\n"+
            "<br >\n"+
            "<p>Last updated %s</p>"%(time.asctime(time.localtime())))
            
            
    
    
    html = []
    for line in template:
        html.append(line)
        if "<!-- TITLE -->" in line:
            html.append(title)
        elif "<!-- NAVBAR -->" in line:
            html.append(navlink)
        elif "<!-- HEADER -->" in line:
            html.append(header)
        elif "<!-- BODY -->" in line:
            html.append(body)
            
    with open("%s"%filename,"w") as htmlf:
        htmlf.write('\n'.join(html))

    