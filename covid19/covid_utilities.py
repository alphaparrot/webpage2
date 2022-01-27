import geopandas as gpd
import pandas as pd
import numpy as np
import datetime
import h5py as h5

def get_tree(group,prefix,searchkey="cases"):
    '''For an HDF5 group, find all subgroups which contain Datasets called searchkey.
    
    The output will be a list of strings which use the given prefix as part of the path.
    
    Parameters
    ----------
    group : h5py.Group
        An h5py Group or open File object
    prefix : str
        String that should be prepended to the output paths
    searchkey : str, optional
        The variable (Dataset name) to look for in subgroups.
        
    Returns
    -------
    list
        List of paths to subgroups within the given group or file, which contain Datasets named searchkey.
    '''
    paths = []
    for k in group:
        if not isinstance(group[k],h5.Dataset):
            newfix = "%s/%s"%(prefix,k)
            if "cases" in group[k]:
                paths.append(newfix)
            paths += get_tree(group[k],newfix) #recurse
    return paths

def unify_files(h5file,shapefile):
    '''Merge an HDF5 covid-19 dataset and a shapefile.
    
    Parameters
    ----------
    h5file : str
        Path to the HDF5 file containing hierarchical COVID-19 data
    shapefile : str
        Path to the unified.shp shapefile (the other files should be in the same directory).
        
    Returns
    -------
    geopandas.GeoDataFrame
        MultiIndexed GeoDataFrame with shape data and coordinates for every COVID-19 record in the HDF5 file.
    '''
    
    hdf = h5.File(h5file,"r")
    tree = get_tree(hdf,"")
    rows = []
    for place in tree:
        if place[0]=="/":
            place = place[1:]
        address = place.split("/")
        addrlabels = [x for x in address]
        while len(addrlabels)<4:
            addrlabels.append('')
        row = {"Path":place}
        if "population" in hdf[place]:
            row["Population"] = hdf[place]["population"][()]
        else:
            row["Population"] = np.nan
        cases = hdf[place]["cases"][:]
        deaths = hdf[place]["deaths"][:]
        rt = hdf[place]["Rt"][:]
        rt2wk = np.convolve(rt,np.ones(14),'valid')/14.0
        drt = np.diff(np.append([0,],rt2wk))
        avgcases = np.convolve(cases,np.ones(7),'valid')/7.0
        avgdeaths = np.convolve(deaths,np.ones(7),'valid')/7.0
        recentcases = np.convolve(cases,np.ones(21),'valid')
        recentdeaths = np.convolve(deaths,np.ones(21),'valid')
        nrows = len(rt2wk)
        dcases = len(cases)-nrows
        ddeaths = len(deaths)-nrows
        dacases = len(avgcases)-nrows
        dadeaths = len(avgdeaths)-nrows
        drcases = len(recentcases)-nrows
        drdeaths = len(recentdeaths)-nrows
        time = np.array((np.datetime64(datetime.datetime.strptime(hdf[place]["latestdate"].asstr()[()],"%a %b %d %Y"))\
                        -np.arange(nrows)[::-1].astype('timedelta64[D]')).tolist())
        for n in range(nrows):
            drow = row.copy()
            drow["Time"] = time[n]
            drow["cases"] = cases[dcases+n]
            drow["cases_avg"] = avgcases[dacases+n]
            drow["deaths"] = deaths[ddeaths+n]
            drow["deaths_avg"] = avgdeaths[dadeaths+n]
            drow["21d_cases"] = recentcases[drcases+n]
            drow["21d_deaths"] = recentdeaths[drdeaths+n]
            drow["Rt"] = rt2wk[n]
            drow["dRt"] = drt[n]
            rows.append(drow)
    covid = gpd.GeoDataFrame(rows)
    hdf.close()
    
    world = gpd.read_file(shapefile)
    
    pandemic = pd.merge(world,covid)
    pandemic.set_index(["Country","State","Region","Area","Time"],inplace=True)
    
    return pandemic
    
    