from scene import Scene
import taichi as ti
from taichi.math import *

exposure = 1.0
PI = 3.1415926535

scene = Scene(voxel_edges = 0, exposure = exposure)
scene.set_floor(-20, (0.6, 0.9, 0.6))
scene.set_directional_light((1, 1, 1), 0.2, vec3(1.0, 1.0, 1.0) / exposure)
scene.set_background_color(vec3(0.5, 0.5, 0.4))

@ti.func
def rotate(pos, rotation):
    return vec3(float(pos.x)*ti.cos(rotation)-float(pos.y)*ti.sin(rotation), 
    float(pos.x)*ti.sin(rotation)+float(pos.y)*ti.cos(rotation), 
    pos.z)

@ti.func
def create_petal(pos, size, src_color, dst_color, tilt, rotation):
    tilt = ti.random(ti.f32) * 0.01 + tilt
    rotation = (ti.random(ti.f32) * 20.0 + rotation) / 360.0 * PI
    for x, y in ti.ndrange((0, size), (0, size)):
        if (x/size)**3+(y/size)**3-(x/size)*(y/size)<=0:
            d = x**2 + y**2
            z = tilt * d
            for offset_x, offset_y, offset_rotate in ti.ndrange((-1, 1), (-1, 1), (0, 6)):
                offset = vec3(offset_x, offset_y, 0)
                offset_rotate_ = PI * float(offset_rotate) / 3.0
                color = src_color
                if d < size ** 2 * 0.6:
                    color = src_color * (1.0 - d / size ** 2 * 0.6) + dst_color
                else:
                    color = dst_color * (1.0 - (d - size ** 2 * 0.6) / size ** 2 * 0.4) + src_color
                scene.set_voxel(pos + offset + rotate(vec3(x, y, z), rotation + offset_rotate_), 1, color)

@ti.func
def create_leave(pos, size, src_color, dst_color, tilt, rotation):
    tilt = ti.random(ti.f32) * 0.01 + tilt
    rotation = (ti.random(ti.f32) * 20.0 + rotation) / 360.0 * PI
    for x, y in ti.ndrange((0, size), (0, size)):
        if (x/size)**3+(y/size)**3-(x/size)*(y/size)<=0:
            d = x**2 + y**2
            z = tilt * d
            for offset_x, offset_y in ti.ndrange((-1, 1), (-1, 1)):
                offset = vec3(offset_x, offset_y, 0)
                color = src_color
                if d < size ** 2 * 0.6:
                    color = src_color * (1.0 - d / size ** 2 * 0.6) + dst_color
                else:
                    color = dst_color * (1.0 - (d - size ** 2 * 0.6) / size ** 2 * 0.4) + src_color
                scene.set_voxel(pos + offset + rotate(vec3(x, y, z), rotation), 1, color)

@ti.func
def create_floor(pos, size, color, color_noise):
    for x, y in ti.ndrange((-size, size), (-size, size)):
        scene.set_voxel(pos + vec3(x, 0, y), 1, color + color_noise * ti.random())

@ti.func
def create_heart(pos, size):
    for x, y in ti.ndrange((-size, size), (-size, size)):
        if x**2 + y ** 2 <= size ** 2:
            scene.set_voxel(pos + vec3(x, y, 0), 1, vec3(1.0, 1.0, 0.0))
            if x % 2 == 1 and y % 2 == 1:
                scene.set_voxel(pos + vec3(x, y, 1), 1, vec3(0.6, 0.6, 0.0))

@ti.func
def create_trunk(pos, radius, height):
    for x, z, y in ti.ndrange((-radius, radius), (-radius, radius), (-height, height)):
        if x**2 + z ** 2 <= radius ** 2:
            scene.set_voxel(pos + vec3(x, y, z), 1, vec3(0.5, 0.25, 0.0))

@ti.func
def jitter(src, scale = vec3(0.1, 0.1, 0.1)):
    src += (ti.random(ti.f32) - 0.5) * scale
    return src

@ti.kernel
def initialize_voxels():
    create_trunk(vec3(0, -30, 2), 3, 30)
    create_petal(vec3(0, 10, 0), 54, jitter(vec3(0.9, 0.9, 0.9)), jitter(vec3(0.9, 0.9, 0.1)), 0.00, 0.0)
    create_petal(vec3(0, 10, 1), 44, jitter(vec3(0.9, 0.9, 0.9)), jitter(vec3(0.9, 0.9, 0.1)), 0.01, 20.0)
    create_petal(vec3(0, 10, 2), 34, jitter(vec3(0.9, 0.9, 0.9)), jitter(vec3(0.9, 0.9, 0.1)), 0.03, 30.0)
    create_petal(vec3(0, 10, 3), 24, jitter(vec3(0.9, 0.9, 0.9)), jitter(vec3(0.9, 0.9, 0.1)), 0.05, 40.0)
    create_heart(vec3(0, 10, 9), 8)

    create_leave(vec3(0, -45, 1), 44, jitter(vec3(0.1, 0.9, 0.1)), jitter(vec3(0.3, 0.9, 0.3)), 0.01, -45.0)
    create_leave(vec3(0, -45, 1), 44, jitter(vec3(0.1, 0.9, 0.1)), jitter(vec3(0.3, 0.9, 0.3)), 0.01, 180.0)

    create_floor(vec3(0, -60, 0), 56, vec3(0.3, 0.9, 0.1), vec3(0.01, 0.08, 0.01))

initialize_voxels()

scene.finish()
