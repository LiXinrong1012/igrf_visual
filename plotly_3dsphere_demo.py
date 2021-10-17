# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 18:03:09 2021

@author: Li Xinrong
"""

import numpy as np           

from mpl_toolkits.basemap import Basemap

from numpy import pi, sin, cos
from ppigrf import igrf,igrf_gc,RE
from datetime import datetime

from plotly.graph_objs import *


import dash
import dash_core_components as dcc
from dash import html

import plotly.express as px

lat=np.arange(-90,90.1,2.5,dtype=np.float32) # start,stop,step
lon=np.arange(-180,180,2.5,dtype=np.float32)
out =  np.zeros((73,144))
date=str(datetime.now()).split('-')
year=int(date[0])
month=int(date[1])
day=int((date[2].split(' '))[0])
date=datetime(year,month,day)
m = Basemap() 

def fig12_draw(h):       
    h = h      # kilometers above sea level    
    data_lat = np.float32([]).reshape(73,0)  #初始化data
    data_lon = np.float32([]).reshape(144,0)
    tmp_lat=lat.reshape(73,1)
    tmp_lon=lon.reshape(144,1)
    for i in range(144):  #想构建144行
        data_lat = np.append(data_lat,tmp_lat,axis=1)
    for i in range(73):  #想构建73行
        data_lon = np.append(data_lon,tmp_lon,axis=1)
    data_lon=data_lon.transpose().reshape(-1,1)
    Be, Bn, Bu = igrf(lon, data_lat, h, date)
    #print(np.max(Be),np.max(Bn),np.max(Bu),np.min(Be),np.min(Bu),np.min(Bn))
    out=np.sqrt(Be**2+Bn**2+Bu**2).reshape(73,144) 
    m = Basemap() 
    #全图的磁场
    cc_lons, cc_lats=get_coastline_traces()
    country_lons, country_lats=get_country_traces()
    
    #concatenate the lon/lat for coastlines and country boundaries:
    lons=cc_lons+[None]+country_lons
    lats=cc_lats+[None]+country_lats

    xs, ys, zs=mapping_map_to_sphere(lons, lats, radius=1.01)# here the radius is slightly greater than 1 
                                                             #to ensure lines visibility; otherwise (with radius=1)
                                                             # some lines are hidden by contours colors
    boundaries=dict(type='scatter3d',
                   x=xs,
                   y=ys,
                   z=zs,
                   mode='lines',
                   line=dict(color='black', width=1)
                  )
    
    colorscale=[[0.0, '#313695'],
     [0.07692307692307693, '#3a67af'],
     [0.15384615384615385, '#5994c5'],
     [0.23076923076923078, '#84bbd8'],
     [0.3076923076923077, '#afdbea'],
     [0.38461538461538464, '#d8eff5'],
     [0.46153846153846156, '#d6ffe1'],
     [0.5384615384615384, '#fef4ac'],
     [0.6153846153846154, '#fed987'],
     [0.6923076923076923, '#fdb264'],
     [0.7692307692307693, '#f78249'],
     [0.8461538461538461, '#e75435'],
     [0.9230769230769231, '#cc2727'],
     [1.0, '#a50026']]
    
    clons=np.array(lon.tolist()+[180], dtype=np.float64)
    clats=np.array(lat, dtype=np.float64)
    clons, clats=np.meshgrid(clons, clats)
    
    XS, YS, ZS=mapping_map_to_sphere(clons, clats)
    
    nrows, ncolumns=clons.shape
    OLR=np.zeros(clons.shape, dtype=np.float64)
    OLR[:, :ncolumns-1]=np.copy(np.array(out,  dtype=np.float64))
    OLR[:, ncolumns-1]=np.copy(out[:, 0])
    

    text=[['lon: '+'{:.2f}'.format(clons[i,j])+'<br>lat: '+'{:.2f}'.format(clats[i, j])+
            '<br>B: '+'{:.2f}'.format(OLR[i][j]) for j in range(ncolumns)] for i in range(nrows)]
    
    sphere=dict(type='surface',
                x=XS, 
                y=YS, 
                z=ZS,
                colorscale=colorscale,
                surfacecolor=OLR,
                cmin=np.min(out), 
                cmax=np.max(out),
                colorbar=dict(thickness=20, len=0.75, ticklen=4, title= 'nT'),
                text=text,
                hoverinfo='text')
    
    noaxis=dict(showbackground=False,
                showgrid=False,
                showline=False,
                showticklabels=False,
                ticks='',
                title='',
                zeroline=False)
    
    layout_3d=dict(title="高度为海平面以上"+str(h)+'km的地球磁场分布'+'<br>'+'（'+str(year)+'年'+str(month)+'月'+str(day)+'日）-球面可视化',
                  font=dict(family='Balto', size=14),
                  width=700, 
                  height=700,
                  scene=dict(xaxis=noaxis, 
                             yaxis=noaxis, 
                             zaxis=noaxis,
                             aspectratio=dict(x=1,
                                              y=1,
                                              z=1),
                             camera=dict(eye=dict(x=1.15, 
                                         y=1.15, 
                                         z=1.15)
                                        )
                ),
               # paper_bgcolor='rgba(235,235,235, 0.9)'  
               )
    
    trace1 = Contour(
        z=out,
        x=lon,
        y=lat,
        colorscale= [[0.0, '#171c42'], [0.07692307692307693, '#263583'], [0.15384615384615385, '#1a58af'], [0.23076923076923078, '#1a7ebd'], [0.3076923076923077, '#619fbc'], [0.38461538461538464, '#9ebdc8'], [0.46153846153846156, '#d2d8dc'], [0.5384615384615384, '#e6d2cf'], [0.6153846153846154, '#daa998'], [0.6923076923076923, '#cc7b60'], [0.7692307692307693, '#b94d36'], [0.8461538461538461, '#9d2127'], [0.9230769230769231, '#6e0e24'], [1.0, '#3c0911']],
        zauto=False,  # custom contour levels
        zmin=-5,      # first contour level
        zmax=5,        # last contour level  => colorscale is centered about 0
          
     
    colorbar= {
        "borderwidth": 0, 
        "outlinewidth": 0, 
        "thickness": 15, 
        "tickfont": {"size": 14}, 
        "title": "nT"}, #gives your legend some units                                                                     
    
    contours= {
        "end": np.max(out), 
        "showlines": False, 
        "size": (np.max(out)-np.min(out))/100, #this is your contour interval
        "start": np.min(out)}     
        
    )    
    # Make shortcut to Basemap object, 
    # not specifying projection type for this example

    # Get list of of coastline, country, and state lon/lat traces
    traces_cc = get_coastline_traces1()+get_country_traces1()
    data_2d = Data([trace1]+traces_cc)
    
    title = "高度为海平面以上"+str(h)+"km的地球磁场分布"+"（"+str(year)+"年"+str(month)+"月"+str(day)+"日）-平面可视化"
    
    axis_style = dict(
        zeroline=False,
        showline=False,
        showgrid=False,
        ticks='',
        showticklabels=False,
    )
    
    layout_2d = Layout(
        title=title,
        
        showlegend=False,
        hovermode="closest",        # highlight closest point on hover
        xaxis=XAxis(
            axis_style,
            range=[lon[0],lon[-1]]  # restrict y-axis to range of lon
        ),
        yaxis=YAxis(
            axis_style,
        ),
        autosize=False,
        width=750,
        height=500,
    )    
    return sphere, boundaries,layout_3d,data_2d,layout_2d


def degree2radians(degree):
    #convert degrees to radians
    return degree*pi/180

def mapping_map_to_sphere(lon, lat, radius=1):
    #this function maps the points of coords (lon, lat) to points onto the  sphere of radius radius
    
    lon=np.array(lon, dtype=np.float64)
    lat=np.array(lat, dtype=np.float64)
    lon=degree2radians(lon)
    lat=degree2radians(lat)
    xs=radius*cos(lon)*cos(lat)
    ys=radius*sin(lon)*cos(lat)
    zs=radius*sin(lat)
    return xs, ys, zs

# Make shortcut to Basemap object, 
# not specifying projection type for this example



# Functions converting coastline/country polygons to lon/lat traces
def polygons_to_traces(poly_paths, N_poly):
    ''' 
    pos arg 1. (poly_paths): paths to polygons
    pos arg 2. (N_poly): number of polygon to convert
    '''
    # init. plotting list
    lons=[]
    lats=[]

    for i_poly in range(N_poly):
        poly_path = poly_paths[i_poly]
        
        # get the Basemap coordinates of each segment
        coords_cc = np.array(
            [(vertex[0],vertex[1]) 
             for (vertex,code) in poly_path.iter_segments(simplify=False)]
        )
        
        # convert coordinates to lon/lat by 'inverting' the Basemap projection
        lon_cc, lat_cc = m(coords_cc[:,0],coords_cc[:,1], inverse=True)
    
        
        lats.extend(lat_cc.tolist()+[None]) 
        lons.extend(lon_cc.tolist()+[None])
        
       
    return lons, lats

def polygons_to_traces1(poly_paths, N_poly):
    ''' 
    pos arg 1. (poly_paths): paths to polygons
    pos arg 2. (N_poly): number of polygon to convert
    '''
    # init. plotting list
    data = dict(
        x=[],
        y=[],
        mode='lines',
        line=Line(color="black"),
        name=' '
    )

    for i_poly in range(N_poly):
        poly_path = poly_paths[i_poly]
        
        # get the Basemap coordinates of each segment
        coords_cc = np.array(
            [(vertex[0],vertex[1]) 
             for (vertex,code) in poly_path.iter_segments(simplify=False)]
        )
        
        # convert coordinates to lon/lat by 'inverting' the Basemap projection
        lon_cc, lat_cc = m(coords_cc[:,0],coords_cc[:,1], inverse=True)
    
        
        # add plot.ly plotting options
        data['x'] = data['x'] + lon_cc.tolist() + [np.nan]
        data['y'] = data['y'] + lat_cc.tolist() + [np.nan]
        
        # traces.append(make_scatter(lon_cc,lat_cc))
     
    return [data]
# Function generating coastline lon/lat 
def get_coastline_traces():
    poly_paths = m.drawcoastlines().get_paths() # coastline polygon paths
    N_poly = 91  # use only the 91st biggest coastlines (i.e. no rivers)
    cc_lons, cc_lats= polygons_to_traces(poly_paths, N_poly)
    return cc_lons, cc_lats
def get_coastline_traces1():
    poly_paths = m.drawcoastlines().get_paths() # coastline polygon paths
    N_poly = 91  # use only the 91st biggest coastlines (i.e. no rivers)
    return polygons_to_traces1(poly_paths, N_poly)

# Function generating country lon/lat 
def get_country_traces():
    poly_paths = m.drawcountries().get_paths() # country polygon paths
    N_poly = len(poly_paths)  # use all countries
    country_lons, country_lats= polygons_to_traces(poly_paths, N_poly)
    return country_lons, country_lats


def get_country_traces1():
    poly_paths = m.drawcountries().get_paths() # country polygon paths
    N_poly = len(poly_paths)  # use all countries
    return polygons_to_traces1(poly_paths, N_poly)
# Get list of of coastline, country, and state lon/lat 

# Make trace-generating function (return a Scatter object)
def make_scatter(x,y):
    return Scatter(
        x=x,
        y=y,
        mode='lines',
        line=Line(color="black"),
        name=' '  # no name on hover
    )


#%%
def fig3_draw(lat):
    N = 200
    phi1=lat/180*np.pi
    phi2=np.pi+phi1

    x = np.linspace(-RE,RE, N)
    y = np.linspace(-RE,RE, N)
    X,Y=np.meshgrid(x,y)
    R = np.sqrt(X**2+Y**2)
    
    Theta=np.abs(np.arcsin(X*(1/R)))

    
   # date = datetime(2021, 3, 28)
    Br,Btheta,Bphi = igrf_gc(R,Theta,phi1,date)
    B1=np.sqrt(Br**2+Btheta**2+Bphi**2).squeeze(0)
    Br,Btheta,Bphi = igrf_gc(R,Theta,phi1,date)
    B2= np.sqrt(Br**2+Btheta**2+Bphi**2).squeeze(0)
    Br,Btheta,Bphi=igrf_gc(R,Theta+np.pi/2,phi2,date)
    B3= np.sqrt(Br**2+Btheta**2+Bphi**2).squeeze(0)
    Br,Btheta,Bphi=igrf_gc(R,Theta+np.pi/2,phi1,date)
    B4= np.sqrt(Br**2+Btheta**2+Bphi**2).squeeze(0)

    B=np.zeros((N,N))
     # mask the outside of the disk of center (0,0)   and radius R
    I,J=np.where((X>=0) & (Y>=0))
    B[I,J]=B1[I,J]
    I,J=np.where((X>=0) & (Y<0))
    B[I,J]=B4[I,J]
    I,J=np.where((X<0) & (Y>=0))
    B[I,J]=B2[I,J]
    I,J=np.where((X<0) & (Y<0))
    B[I,J]=B3[I,J]
    I, J = np.where(R>RE)
    B[I, J] = None

    colorscale=[[0.0, '#313695'],
     [B[3*N//100,62*N//100]/B[49*N//100,49*N//100], '#3a67af'],
     [B[9*N//100,62*N//100]/B[49*N//100,49*N//100], '#5994c5'],
     [B[15*N//100,62*N//100]/B[49*N//100,49*N//100], '#84bbd8'],
     [B[20*N//100,62*N//100]/B[49*N//100,49*N//100],'#afdbea'],
     [B[25*N//100,62*N//100]/B[49*N//100,49*N//100], '#d8eff5'],
     [B[27*N//100,62*N//100]/B[49*N//100,49*N//100], '#d6ffe1'],
     [B[32*N//100,62*N//100]/B[49*N//100,49*N//100],'#fef4ac'],
     [B[36*N//100,62*N//100]/B[49*N//100,49*N//100], '#fed987'],
     [B[42*N//100,62*N//100]/B[49*N//100,49*N//100],'#fdb264'],
     [B[48*N//100,62*N//100]/B[49*N//100,49*N//100],'#f78249'],
     [B[56*N//100,48*N//100]/B[49*N//100,49*N//100], '#e75435'],
     [B[51*N//100,48*N//100]/B[49*N//100,49*N//100], '#cc2727'],
     [1.0, '#a50026']]
    trace = dict(type='heatmap',
              x=x, 
              y=y, 
              z=B, #note that z has the shape of X,Y, not x, y as in your example!!!!!!!!!!
              colorscale=colorscale, 
              showscale=True,
              colorbar=dict(thickness=20, len=0.75, ticklen=4, title= 'nT')
              )
    
    
    layout = dict(title="经度为东经"+str(lat)+"度的地球子午面磁场分布"+ '<br>'+"（"+str(year)+"年"+str(month)+"月"+str(day)+"日）-平面可视化",
                  width=600,
                  height=600,
                  showlegend=False,
                  xaxis=dict(visible=False),
                  yaxis=dict(visible=False)
                 )
    return trace,layout

sphere, boundaries,layout_3d,data_2d,layout_2d=fig12_draw(h=0)
trace,layout=fig3_draw(lat=0)  
fig_2d = Figure(data=data_2d, layout=layout_2d)

fig=dict(data=[sphere, boundaries], layout=layout_3d)
fig_2d_polar = Figure(data=trace, layout=layout)
    
#%%
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(children=[
    html.H1(children="航天器系统设计作业",
            style={
            'textAlign': 'center',
            #'color': colors['text']
        }),
    html.H2(children="基于IGRF模型的地球磁场可视化",style={
            'textAlign': 'center'}),
    html.Div(children='''
        力9 李欣荣 2019011480（完成日期：2021.10.15）
    ''',style={
            'textAlign': 'center',
           # 'color': colors['text']
        }),       

   # dcc.Input(value='MTL', type='text'),
    dcc.Graph(id='3d',
              figure=fig,
              style={'float': 'left','margin': 'auto'}),
    html.Div(children=[
    html.H4("输入海平面以上高度h/km或输入子午面经度（东经）lat/度"),
    html.Div([
        "海平面以上高度：",
        dcc.Input(id='my-input-h', value=0, type='number')
    ]),
    html.Br()]),
  #  html.Div(id='my-output')]),
    html.Div([
        "子午面经度（东经）: ",
        dcc.Input(id='my-input-lon', value=0, type='number')
    ]),
 
    dcc.Graph(id='2d',
              figure=fig_2d,
              style={'float': 'Left','margin': 'auto'}),
 
    dcc.Graph(id='polar',
              figure=fig_2d_polar,
              style={'float': 'Right','margin': 'auto'}),

])
  


@app.callback(
    Output(component_id='3d', component_property='figure'),
    Output(component_id='2d', component_property='figure'),
    Input(component_id='my-input-h', component_property='value'))
def update_figure(h):
    sphere, boundaries,layout_3d,data_2d,layout_2d=fig12_draw(h)
    fig_2d = Figure(data=data_2d, layout=layout_2d)
    fig=dict(data=[sphere, boundaries], layout=layout_3d)
 #   fig.update_layout(transition_duration=500)
   # fig_2d.update_layout(transition_duration=500)  
    return fig,fig_2d

@app.callback(
    Output(component_id='polar', component_property='figure'),    
    Input(component_id='my-input-lon', component_property='value'))
def update_figure_polar(lat):
    trace,layout=fig3_draw(lat)  
    fig_2d_polar = Figure(data=trace, layout=layout)
    
 #   fig.update_layout(transition_duration=500)
   # fig_2d.update_layout(transition_duration=500)  
    return fig_2d_polar

"""
@app.callback(
    Output('3d', 'figure'),
    Input('my-input-h', 'value'))
def update_figure1(h):
    
    
    fig.update_layout(transition_duration=500)

    return fig
"""

if __name__ == '__main__':
    app.run_server(debug=True)
















