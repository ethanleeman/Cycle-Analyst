import pandas as pd
import shapely.geometry as geometry


def clean_crashes(df,polygon,north=None,south=None,east=None,west=None):
    an = df.copy()
    an = an[an['BICYCLE_COUNT'] > 0]
    in_location = an.apply(lambda r: polygon.contains(geometry.Point(r['DEC_LONG'], r['DEC_LAT'])),axis=1)
    an = an[in_location]
    if north is not None:
        an = an[an['DEC_LONG'] < east]
        an = an[an['DEC_LONG'] > west]
        an = an[an['DEC_LAT'] < north]
        an = an[an['DEC_LAT'] > south]
    return an

def clean_traffic(df,polygon,north=None,south=None,east=None,west=None):
    an = df.copy()
    if north is not None:
        an = an[an['X'] < east]
        an = an[an['X'] > west]
        an = an[an['Y'] < north]
        an = an[an['Y'] > south]
    in_location = an.apply(lambda r: polygon.contains(geometry.Point(r['X'], r['Y'])),axis=1)
    an = an[in_location]
    return an

def exploder_one_hot(df,column_name_list):
    to_concat = [df.copy()]
    for col in column_name_list:
        to_concat.append(df[col].apply(lambda x: [x]).str.join('|').str.get_dummies().add_prefix(col + ":"))
        #return df[col].apply(lambda x: [x]).str.join('|').str.get_dummies()
    return pd.concat(to_concat,axis=1).drop(column_name_list,axis=1)

def edge_featurizer(df,column_name_list):
    an = exploder_one_hot(df,column_name_list)
    an['x'] = an.apply(lambda r: r.geometry.centroid.x, axis=1)
    an['y'] = an.apply(lambda r: r.geometry.centroid.y, axis=1)
    an['oneway'] = an['oneway']*1.0
    return an
