#!/usr/bin/env python3
"""Original brand-neutral mobile AMC concept, SI units, base footprint at floor."""
from pathlib import Path
import xml.etree.ElementTree as E

ROOT=Path(__file__).resolve().parents[1]
r=E.Element('robot',name='mobile_amc')
def e(p,t,**kw): return E.SubElement(p,t,{k:str(v) for k,v in kw.items()})
def text(p,t,v): n=e(p,t);n.text=str(v);return n
def origin(p,xyz,rpy='0 0 0'): e(p,'origin',xyz=xyz,rpy=rpy)
def box(name,xyz,size,color,mass=0,collision=True,parent='base_footprint'):
    l=e(r,'link',name=name)
    for kind in ['visual','collision'] if collision else ['visual']:
        v=e(l,kind);e(e(v,'geometry'),'box',size=size)
        if kind=='visual': e(e(v,'material',name=name+'_color'),'color',rgba=color)
    if mass:
        a,b,c=map(float,size.split());i=e(l,'inertial');e(i,'mass',value=mass)
        e(i,'inertia',ixx=mass*(b*b+c*c)/12,iyy=mass*(a*a+c*c)/12,izz=mass*(a*a+b*b)/12,ixy=0,ixz=0,iyz=0)
    j=e(r,'joint',name=name+'_fixed',type='fixed');e(j,'parent',link=parent);e(j,'child',link=name);origin(j,xyz)
    g=e(r,'gazebo',reference=name);text(g,'material',{'white':'Gazebo/White'}.get(color,'Gazebo/White'))
    return l
def frame(name,xyz):
    e(r,'link',name=name);j=e(r,'joint',name=name+'_fixed',type='fixed');e(j,'parent',link='base_footprint');e(j,'child',link=name);origin(j,xyz)

e(r,'link',name='base_footprint')
box('base_link','0 0 0.22','.84 .50 .24','.14 .20 .24 1',40)
box('analyzer_cabinet','-.02 0 .76','.66 .48 .84','.91 .94 .95 1',20)
box('bumper_front','.43 0 .18','.04 .60 .11','.08 .12 .15 1',.4)
box('bumper_back','-.43 0 .18','.04 .60 .11','.08 .12 .15 1',.4)
box('front_panel','.315 0 .78','.014 .39 .68','.12 .22 .28 1',collision=False)
box('display','.327 0 1.00','.012 .26 .16','.05 .62 .72 1',collision=False)
box('filter_cartridge','.325 0 .67','.02 .34 .32','.45 .56 .61 1',collision=False)
for i in range(7): box(f'intake_grille_{i}',f'.34 0 {.54+i*.032}', '.016 .30 .012','.05 .1 .13 1',collision=False)
box('molecular_filter_indicator','.329 -.13 .90','.018 .03 .03','.2 .8 .45 1',collision=False)
box('sampling_mast','.24 .16 1.24','.035 .035 .24','.52 .62 .66 1',.2)
box('sampling_inlet','.345 .16 1.35','.21 .045 .045','.09 .55 .66 1',.1)
box('rear_exhaust','-.357 0 .84','.015 .32 .25','.1 .24 .3 1',collision=False)
for i in range(6): box(f'exhaust_grille_{i}',f'-.37 0 {.745+i*.035}', '.013 .27 .012','.52 .63 .67 1',collision=False)
box('beacon','-.24 0 1.21','.06 .06 .07','.05 .8 .6 1',.1)
box('sampling_inlet_low','.40 .16 .45','.10 .04 .04','.09 .55 .66 1',.05)
frame('gas_sensor_link','.45 .16 .45')
frame('gas_sensor_high_link','.45 .16 1.35')
frame('purifier_inlet_link','.38 0 .64')
frame('purifier_outlet_link','-.38 0 .84')

for side,y in [('left',.28),('right',-.28)]:
    name='wheel_'+side+'_link';l=e(r,'link',name=name)
    for kind in ['visual','collision']:
        v=e(l,kind);origin(v,'0 0 0','1.57079632679 0 0');e(e(v,'geometry'),'cylinder',radius=.10,length=.07)
        if kind=='visual':e(e(v,'material',name='tire_'+side),'color',rgba='.05 .06 .07 1')
    i=e(l,'inertial');e(i,'mass',value=2);e(i,'inertia',ixx=.0058,iyy=.01,izz=.0058,ixy=0,ixz=0,iyz=0)
    j=e(r,'joint',name='wheel_'+side+'_joint',type='continuous');e(j,'parent',link='base_footprint');e(j,'child',link=name)
    origin(j,f'0 {y} .1');e(j,'axis',xyz='0 1 0');e(j,'dynamics',damping=.05,friction=0)
    g=e(r,'gazebo',reference=name);text(g,'mu1',1);text(g,'mu2',1);text(g,'material','Gazebo/Black')
for name,x in [('front',.32),('rear',-.32)]:
    l=e(r,'link',name='caster_'+name)
    for kind in ['visual','collision']: e(e(e(l,kind),'geometry'),'sphere',radius=.04)
    i=e(l,'inertial');e(i,'mass',value=.2);e(i,'inertia',ixx=.000128,iyy=.000128,izz=.000128,ixy=0,ixz=0,iyz=0)
    j=e(r,'joint',name='caster_'+name+'_fixed',type='fixed');e(j,'parent',link='base_footprint');e(j,'child',link='caster_'+name);origin(j,f'{x} 0 .04')
    g=e(r,'gazebo',reference='caster_'+name);text(g,'mu1',.001);text(g,'mu2',.001)

box('base_scan','.37 0 .36','.10 .10 .05','.08 .12 .14 1',.15)
g=e(r,'gazebo',reference='base_scan');s=e(g,'sensor',name='amc_lidar',type='ray')
text(s,'always_on','true');text(s,'update_rate',10);text(s,'visualize','false')
ray=e(s,'ray');h=e(e(ray,'scan'),'horizontal')
for k,v in [('samples',720),('resolution',1),('min_angle',-3.14159265),('max_angle',3.14159265)]: text(h,k,v)
range_=e(ray,'range')
for k,v in [('min',.12),('max',12),('resolution',.01)]:text(range_,k,v)
p=e(s,'plugin',name='amc_lidar_ros',filename='libgazebo_ros_ray_sensor.so');text(e(p,'ros'),'remapping','~/out:=scan');text(p,'output_type','sensor_msgs/LaserScan');text(p,'frame_name','base_scan')
g=e(r,'gazebo');p=e(g,'plugin',name='amc_diff_drive',filename='libgazebo_ros_diff_drive.so')
for k,v in [('update_rate',50),('left_joint','wheel_left_joint'),('right_joint','wheel_right_joint'),('wheel_separation',.56),('wheel_diameter',.2),('max_wheel_torque',35),('max_wheel_acceleration',.6),('publish_odom','true'),('publish_odom_tf','true'),('publish_wheel_tf','false'),('odometry_frame','odom'),('robot_base_frame','base_footprint'),('odometry_source',0)]:text(p,k,v)
p=e(g,'plugin',name='amc_joint_states',filename='libgazebo_ros_joint_state_publisher.so')
text(p,'update_rate',30);text(p,'joint_name','wheel_left_joint');text(p,'joint_name','wheel_right_joint')
# Set materials explicitly for Classic as URDF colour alone is not always retained.
palette={'base_link':'Gazebo/DarkGrey','front_panel':'Gazebo/DarkGrey','display':'Gazebo/Turquoise','filter_cartridge':'Gazebo/Grey','sampling_mast':'Gazebo/Grey','sampling_inlet':'Gazebo/Turquoise','beacon':'Gazebo/Green','rear_exhaust':'Gazebo/DarkGrey','base_scan':'Gazebo/Black','bumper_front':'Gazebo/Black','bumper_back':'Gazebo/Black'}
for g in r.findall('gazebo'):
    name=g.get('reference','');m=g.find('material')
    if m is not None:
        if 'grille' in name:m.text='Gazebo/DarkGrey'
        elif name in palette:m.text=palette[name]
E.indent(r)
(ROOT/'urdf').mkdir(exist_ok=True)
E.ElementTree(r).write(ROOT/'urdf/mobile_amc.urdf',encoding='utf-8',xml_declaration=True)
print('Generated mobile_amc.urdf')
