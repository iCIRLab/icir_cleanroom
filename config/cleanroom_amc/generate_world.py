#!/usr/bin/env python3
"""Single geometry source for Gazebo, navigation and simplified gas transport."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WHITE = '0.85 0.9 0.93 1'
STEEL = '0.57 0.66 0.7 1'
BLUE = '0.07 0.29 0.39 1'
CYAN = '0.15 0.8 0.9 1'
OBSTACLES = []


def tag(parent, element, text=None, **attrs):
    n = E.SubElement(parent, element, attrs)
    if text is not None:
        n.text = str(text)
    return n


def shape(link, name, pos, size, color, collision=True, transparency=0, rpy=(0, 0, 0)):
    for typ in (['collision', 'visual'] if collision else ['visual']):
        e = tag(link, typ, name=name)
        tag(e, 'pose', ' '.join(map(str, (*pos, *rpy))))
        tag(tag(tag(e, 'geometry'), 'box'), 'size', ' '.join(map(str, size)))
        if typ == 'visual':
            m = tag(e, 'material')
            tag(m, 'ambient', color)
            tag(m, 'diffuse', color)
            tag(e, 'transparency', transparency)
            tag(e, 'cast_shadows', 'false')
    return e


def model(world, name):
    m = tag(world, 'model', name=name)
    tag(m, 'static', 'true')
    return tag(m, 'link', name='body')


def obstacle(world, name, xyz, size, color=WHITE, cut=False):
    OBSTACLES.append(dict(name=name, center=list(xyz), size=list(size)))
    l = model(world, name)
    shape(l, 'body', xyz, size, color, transparency=0.88 if cut else 0)
    return l


def make_world(cutaway):
    OBSTACLES.clear()
    root = E.Element('sdf', version='1.6')
    w = tag(root, 'world', name='cleanroom')
    tag(w, 'gravity', '0 0 -9.81')
    p = tag(w, 'physics', name='ode', type='ode')
    tag(p, 'max_step_size', '0.001')
    tag(p, 'real_time_update_rate', '1000')
    scene = tag(w, 'scene')
    tag(scene, 'ambient', '0.7 0.7 0.7 1')
    tag(scene, 'background', '0.08 0.12 0.17 1')
    tag(scene, 'shadows', 'false')
    light = tag(w, 'light', name='room_light', type='directional')
    tag(light, 'pose', '0 0 8 0 0 0')
    tag(light, 'diffuse', '0.9 0.9 0.9 1')
    tag(light, 'direction', '0.2 0.1 -1')
    camera = tag(tag(w, 'gui', fullscreen='0'), 'camera', name='overview')
    tag(camera, 'pose', '19 -24 23 0 0.63 2.23')
    plugin = tag(w, 'plugin', name='gazebo_ros_state', filename='libgazebo_ros_state.so')
    tag(tag(plugin, 'ros'), 'namespace', '/gazebo')
    tag(plugin, 'update_rate', 20)

    floor = model(w, 'raised_floor')
    shape(floor, 'walkable_surface', (0, 0, -0.12), (20.4, 14.4, 0.24), '0.62 0.72 0.75 1')
    # Grating is visual: continuous collision plane prevents tiny wheels catching.
    for ix in range(20):
        for iy in range(14):
            x, y = ix-9.5, iy-6.5
            shape(floor, f'tile_{ix}_{iy}', (x, y, 0.002), (.975, .975, .003), '0.77 0.83 0.84 1', False)
            for j in range(4):
                shape(floor, f'grille_{ix}_{iy}_{j}', (x, y-.3+.2*j, .004), (.68, .022, .002), STEEL, False)
    for name, xyz, size in [
        ('wall_north', (0, 7.1, 1.65), (20.4, .2, 3.3)),
        ('wall_south', (0, -7.1, 1.65), (20.4, .2, 3.3)),
        ('wall_east', (10.1, 0, 1.65), (.2, 14, 3.3)),
        ('wall_west', (-10.1, 0, 1.65), (.2, 14, 3.3)),
        ('airlock_north', (-8.25, -3, 1.65), (3.5, .15, 3.3)),
        ('airlock_door_south', (-6.5, -6.4, 1.65), (.15, 1.2, 3.3)),
        ('airlock_door_north', (-6.5, -3.6, 1.65), (.15, 1.2, 3.3)),
    ]:
        obstacle(w, name, xyz, size, cut=cutaway and name in ['wall_south', 'wall_east'])
    roof = model(w, 'ceiling_ffu_array')
    shape(roof, 'ceiling', (0, 0, 3.45), (20.4, 14.4, .2), WHITE, transparency=.98 if cutaway else 0)
    # Ceiling structure remains physically present in cutaway view.
    for ix in range(10):
        for iy in range(7):
            x, y = -9+2*ix, -6+2*iy
            shape(roof, f'ffu_frame_{ix}_{iy}', (x, y, 3.28), (1.82, 1.82, .12), STEEL, False, .95 if cutaway else 0)
            shape(roof, f'hepa_{ix}_{iy}', (x, y, 3.21), (1.65, 1.65, .02), WHITE, False, .95 if cutaway else 0)
    service = model(w, 'service_details')
    for x in [-8, -4, 0, 4, 8]:
        shape(service, f'return_{x}', (x, 6.98, .17), (1.3, .02, .25), BLUE, False)
    for x in [-4, 1, 6]:
        shape(service, f'utility_header_{x}', (x, 6.8, 2.8), (.07, .07, .8), CYAN, False)
    shape(service, 'air_shower_mat', (-8, -5, .008), (2, 2, .012), '0.05 0.4 0.53 1', False)
    for i in range(4):
        shape(service, f'air_shower_nozzle_{i}', (-9.96, -5.6+.4*i, 1.3), (.025, .12, .12), BLUE, False)

    machines = [
        ('etch_A', (-3, 3, 1.15), (2.4, 2, 2.3)),
        ('CVD_B', (2, 3, 1.2), (2.4, 2, 2.4)),
        ('lithography_C', (6.3, 3, 1.15), (2.6, 2.2, 2.3)),
        ('wet_bench_D', (-3, -3, .95), (2.4, 2, 1.9)),
        ('metrology_E', (2, -3, .95), (2.4, 2, 1.9)),
        ('process_F', (6.3, -3, 1.1), (2.6, 2.2, 2.2)),
        ('gas_cabinet_1', (-5, 6.35, 1.1), (1.2, .9, 2.2)),
        ('gas_cabinet_2', (-3.2, 6.35, 1.1), (1.2, .9, 2.2)),
        ('gas_cabinet_3', (-1.4, 6.35, 1.1), (1.2, .9, 2.2)),
        ('clean_bench', (-8.8, 2, .8), (1.4, 3, 1.6)),
        ('pass_box', (-8.8, 5.5, .75), (1.4, 1.4, 1.5)),
        ('utility_rack', (9.25, 5.5, 1), (1, 2, 2)),
    ]
    for name, pos, size in machines:
        l = obstacle(w, name, pos, size, STEEL if 'cabinet' in name else WHITE)
        x,y,z = pos
        sx,sy,sz = size
        front = y-sy/2-.012
        shape(l, 'front_panel', (x, front, .52*sz), (.82*sx, .018, .60*sz), BLUE, False)
        shape(l, 'window', (x-.12*sx, front-.012, .68*sz), (.45*sx, .018, .19*sz), '0.15 0.52 0.66 1', False)
        shape(l, 'control_panel', (x+.31*sx, front-.02, .67*sz), (.18*sx, .02, .22*sz), '0.06 0.1 0.13 1', False)
        shape(l, 'status_green', (x+.35*sx, y, sz+.045), (.06,.06,.08), '0.1 0.9 0.5 1', False)
        shape(l, 'base_plinth', (x, front, .12), (.9*sx, .02, .17), STEEL, False)
        if 'cabinet' in name:
            shape(l, 'hazard_plate', (x, front-.025, 1.7), (.25,.01,.20), '1 .74 .12 1', False)
    if cutaway:
        arrows = model(w, 'downflow_visual_only')
        for i,(x,y) in enumerate([(-5,0),(0,0),(4,0),(8.3,0),(-5,4.8),(0,4.8),(4,4.8),(8.3,-5.5)]):
            shape(arrows, f'shaft_{i}', (x,y,2.3), (.025,.025,.7), CYAN, False, .45)
            for k in [-1,1]:
                shape(arrows, f'head_{i}_{k}', (x+k*.08,y,2), (.025,.025,.23), CYAN, False, .45, (0,k*.75,0))
    E.indent(root)
    file = ROOT/'worlds'/('cleanroom.world' if cutaway else 'cleanroom_enclosed.world')
    E.ElementTree(root).write(file, encoding='utf-8', xml_declaration=True)
    return list(OBSTACLES)


def generate():
    make_world(False)
    obstacles = make_world(True)
    sources = [
        ('S01_etch', [-3,1.7,.7]), ('S02_cvd', [2,1.7,1.2]),
        ('S03_lithography', [6.3,1.6,.5]), ('S04_wet_bench', [-3,-1.7,.6]),
        ('S05_metrology', [2,-1.7,.8]), ('S06_process', [6.3,-1.6,1.2]),
        ('S07_cabinet', [-5,5.6,1.0]), ('S08_cabinet', [-3.2,5.6,.6]),
        ('S09_cabinet', [-1.4,5.6,1.4]), ('S10_bench', [-7.8,2,.8]),
        ('S11_passbox', [-7.8,5.5,.4]), ('S12_utility', [8.4,5.5,1.2]),
    ]
    scene = dict(room_size=[20,14,3.2], spawn=[-8,-5,0.01],
                 obstacles=obstacles, sources=[dict(name=n, position=p) for n,p in sources])
    (ROOT/'config/geometry.json').write_text(json.dumps(scene, indent=2)+'\n')
    # Cell centres and y-flip match Nav2's lower-left map origin convention.
    resolution=.05
    xs=-10.5+(np.arange(420)+.5)*resolution
    ys=-7.5+(np.arange(300)+.5)*resolution
    xx,yy=np.meshgrid(xs,ys)
    grid=np.full(xx.shape, 254, dtype=np.uint8)
    grid[(abs(xx)>10)|(abs(yy)>7)]=205
    for o in obstacles:
        x,y,z=o['center']; sx,sy,sz=o['size']
        grid[(abs(xx-x)<=sx/2)&(abs(yy-y)<=sy/2)]=0
    Image.fromarray(grid[::-1]).save(ROOT/'maps/cleanroom.pgm')
    (ROOT/'maps/cleanroom.yaml').write_text('image: cleanroom.pgm\nmode: trinary\nresolution: 0.05\norigin: [-10.5, -7.5, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    # Original geometry in OBJ for inspection / CFD preprocessing, not a ready CFD mesh.
    vertices=[]; faces=[]
    for o in obstacles:
        c=np.array(o['center']); s=np.array(o['size'])/2
        base=len(vertices)+1
        vertices.extend([c+s*np.array(v) for v in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]])
        for face in [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]:
            faces.append([base+i for i in face])
    (ROOT/'worlds/obstacles.obj').write_text('# metres; world origin; walls/equipment only; no floor or ceiling\n'+'\n'.join('v '+' '.join(map(str,v)) for v in vertices)+'\n'+'\n'.join('f '+' '.join(map(str,f)) for f in faces)+'\n')
    print(f'Generated 2 worlds, map, {len(obstacles)} collision boxes, {len(sources)} source sites.')


if __name__=='__main__':
    generate()
