"""PROJECT EVENT HORIZON | CINEMATIC EDITION

A self-contained, procedural 3D space observatory built with Pygame + ModernGL.

Physics (CPU / NumPy / RK4):
  * Schwarzschild null geodesics on a fixed black-hole background.
  * Ellis-Bronnikov null geodesics on a separate fixed wormhole background.

Rendering (OpenGL 3.3+):
  * Procedural space / nebula background shader.
  * Flamm paraboloid and Ellis catenoid spatial embedding surfaces.
  * Procedural accretion-disk mesh, horizon visual, photon ring, wormhole throat.
  * Animated geodesic markers, emissive materials, bloom and filmic vignette.
  * Adaptive internal render resolution (dynamic resolution scaling).

Scientific boundary:
  The black hole and wormhole remain two independent exact fixed metrics.
  This software does not claim to simulate their collision or evolve Einstein's
  field equations dynamically. Embedding height is a visualization of a spatial
  slice, not a literal extra spatial dimension. The accretion disk is illustrative.

Run:
  py -m pip install -r requirements.txt
  py event_horizon_cinematic.py
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pygame
import moderngl

from event_horizon_physics import (
    ModelParameters,
    Geodesic,
    build_geodesics,
    schwarzschild_embedding,
    ellis_embedding,
)

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
APP_NAME = "PROJECT EVENT HORIZON | CINEMATIC EDITION"
DEFAULT_WINDOW = (1600, 900)
MIN_RENDER_SCALE = 0.55
MAX_RENDER_SCALE = 1.00
TARGET_FRAME_MS = 16.7

BH_CENTER = np.array([-15.0, 0.0, 0.0], dtype=np.float64)
WH_CENTER = np.array([15.0, 0.0, 0.0], dtype=np.float64)
BH_R_MAX_FACTOR = 12.0
WH_L_MAX_FACTOR = 6.0

# Deep-space palette.
SPACE_DARK = (0.008, 0.012, 0.030)
CYAN = (0.22, 0.86, 1.00)
VIOLET = (0.65, 0.40, 1.00)
GOLD = (1.00, 0.66, 0.25)

# Vertex format: position(3), normal(3), color(3), uv(2) = 11 floats.
MESH_VERTEX_SHADER = """#version 330
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in vec2 in_uv;
uniform mat4 u_mvp;
uniform float u_point_size;
out vec3 v_normal;
out vec3 v_color;
out vec2 v_uv;
out float v_distance;
uniform vec3 u_camera;
void main() {
    vec4 world = vec4(in_position, 1.0);
    gl_Position = u_mvp * world;
    gl_PointSize = u_point_size;
    v_normal = in_normal;
    v_color = in_color;
    v_uv = in_uv;
    v_distance = distance(in_position, u_camera);
}
"""

MESH_FRAGMENT_SHADER = """#version 330
in vec3 v_normal;
in vec3 v_color;
in vec2 v_uv;
in float v_distance;
uniform float u_time;
uniform float u_emission;
uniform int u_style;
uniform int u_draw_point;
out vec4 frag_color;
void main() {
    if (u_draw_point == 1 && length(gl_PointCoord - vec2(0.5)) > 0.5)
        discard;
    vec3 n = normalize(v_normal);
    vec3 light_dir = normalize(vec3(-0.45, -0.55, 0.72));
    float diffuse = max(dot(n, light_dir), 0.0);
    float light = 0.30 + 0.70 * diffuse;
    vec3 base = v_color * light;
    if (u_style == 1) {
        float swirl = sin(v_uv.y * 42.0 - v_uv.x * 13.0 - u_time * 1.35
                          + 2.0 * sin(v_uv.y * 11.0 + u_time * 0.22));
        float turbulence = 0.82 + 0.18 * swirl;
        float inner_heat = 0.88 + 0.12 * (1.0 - v_uv.x);
        base *= turbulence * inner_heat;
        base += vec3(0.22, 0.075, 0.018) * max(swirl, 0.0);
    }
    base += v_color * u_emission;
    float fog = smoothstep(105.0, 235.0, v_distance);
    vec3 space_color = vec3(0.012, 0.020, 0.050);
    base = mix(base, space_color, fog * 0.48);
    frag_color = vec4(base, 1.0);
}
"""

QUAD_VERTEX_SHADER = """#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

BACKGROUND_FRAGMENT_SHADER = """#version 330
in vec2 v_uv;
uniform vec2 u_resolution;
uniform float u_time;
uniform vec2 u_camera_offset;
out vec4 frag_color;
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float valueNoise2D(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.52;
    for (int i = 0; i < 4; ++i) {
        value += amplitude * valueNoise2D(p);
        p = p * 2.03 + vec2(7.1, 3.7);
        amplitude *= 0.5;
    }
    return value;
}
void main() {
    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    vec2 p = (v_uv - 0.5) * vec2(aspect, 1.0);
    float angle = u_camera_offset.x * 0.12;
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    p = rot * p;
    p += vec2(u_camera_offset.y * 0.08, u_camera_offset.x * 0.035);
    vec2 drift = vec2(u_time * 0.004, -u_time * 0.002);
    float warp = fbm(p * 2.1 + drift);
    float clouds = fbm(p * 3.4 + vec2(warp * 1.6, -warp * 1.2) - drift * 0.7);
    float cloud_mask = smoothstep(0.40, 0.79, clouds + warp * 0.16);
    float filament = pow(max(0.0, fbm(p * 7.2 + warp * 2.8) - 0.48), 2.0) * 2.7;
    vec3 col = vec3(0.006, 0.010, 0.027);
    vec3 indigo = vec3(0.055, 0.045, 0.17);
    vec3 violet = vec3(0.20, 0.065, 0.30);
    vec3 teal = vec3(0.015, 0.16, 0.22);
    col += mix(indigo, violet, smoothstep(0.28, 0.78, warp)) * cloud_mask * 0.78;
    col += teal * smoothstep(0.54, 0.82, warp * 0.62 + clouds * 0.38) * 0.42;
    col += vec3(0.16, 0.12, 0.32) * filament * cloud_mask;
    // Fine, deterministic star cells with sub-pixel twinkle.
    vec2 cell = floor(v_uv * vec2(320.0, 190.0));
    float rnd = hash21(cell + 13.7);
    vec2 local = fract(v_uv * vec2(320.0, 190.0)) - 0.5;
    float star_shape = 1.0 - smoothstep(0.015, 0.105, length(local));
    float twinkle = 0.72 + 0.28 * sin(u_time * (1.0 + rnd * 3.0) + rnd * 60.0);
    float star = step(0.986, rnd) * star_shape * twinkle;
    float bright_star = step(0.9985, rnd) * star_shape;
    col += vec3(0.52, 0.66, 1.0) * star * 0.70;
    col += vec3(0.72, 0.85, 1.0) * bright_star * 1.8;
    float vignette = 1.0 - smoothstep(0.18, 1.12, length((v_uv - 0.5) * vec2(aspect, 1.0)));
    col *= mix(0.62, 1.0, vignette);
    frag_color = vec4(col, 1.0);
}
"""

BRIGHT_FRAGMENT_SHADER = """#version 330
in vec2 v_uv;
uniform sampler2D u_scene;
out vec4 frag_color;
void main() {
    vec3 c = texture(u_scene, v_uv).rgb;
    float l = max(max(c.r, c.g), c.b);
    float contribution = smoothstep(0.62, 1.30, l);
    frag_color = vec4(c * contribution, 1.0);
}
"""

BLUR_FRAGMENT_SHADER = """#version 330
in vec2 v_uv;
uniform sampler2D u_image;
uniform vec2 u_direction;
uniform vec2 u_texel;
out vec4 frag_color;
void main() {
    vec2 step_uv = u_direction * u_texel;
    vec3 result = texture(u_image, v_uv).rgb * 0.227027;
    result += texture(u_image, v_uv + step_uv * 1.384615).rgb * 0.1945946;
    result += texture(u_image, v_uv - step_uv * 1.384615).rgb * 0.1945946;
    result += texture(u_image, v_uv + step_uv * 3.230769).rgb * 0.1216216;
    result += texture(u_image, v_uv - step_uv * 3.230769).rgb * 0.1216216;
    result += texture(u_image, v_uv + step_uv * 5.0).rgb * 0.0700;
    result += texture(u_image, v_uv - step_uv * 5.0).rgb * 0.0700;
    frag_color = vec4(result, 1.0);
}
"""

FINAL_FRAGMENT_SHADER = """#version 330
in vec2 v_uv;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_strength;
uniform float u_time;
out vec4 frag_color;
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;
    vec3 hdr = scene + bloom * u_bloom_strength;
    // Soft filmic tone mapping and gentle grain.
    vec3 mapped = hdr / (1.0 + hdr);
    float grain = (hash21(v_uv * 1700.0 + u_time) - 0.5) * 0.012;
    mapped += grain;
    float vignette = 1.0 - smoothstep(0.18, 1.08, length((v_uv - 0.5) * vec2(1.78, 1.0)));
    mapped *= mix(0.72, 1.0, vignette);
    mapped = pow(max(mapped, 0.0), vec3(0.92));
    frag_color = vec4(mapped, 1.0);
}
"""

OVERLAY_FRAGMENT_SHADER = """#version 330
in vec2 v_uv;
uniform sampler2D u_overlay;
out vec4 frag_color;
void main() {
    frag_color = texture(u_overlay, v_uv);
}
"""


@dataclass
class RenderItem:
    vao: moderngl.VertexArray
    count: int
    mode: int
    group: str
    emission: float = 0.0
    style: int = 0


@dataclass
class RayRenderItem:
    geodesic: Geodesic
    item: RenderItem


@dataclass
class SceneResources:
    items: list[RenderItem]
    ray_items: list[RayRenderItem]
    marker_buffer: moderngl.Buffer
    marker_vao: moderngl.VertexArray
    marker_capacity: int

    def release(self) -> None:
        self.marker_vao.release()
        self.marker_buffer.release()
        for item in self.items:
            item.vao.release()
        for ray in self.ray_items:
            ray.item.vao.release()


# -----------------------------------------------------------------------------
# Camera and matrices
# -----------------------------------------------------------------------------
def perspective(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
    mat = np.zeros((4, 4), dtype=np.float64)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= max(np.linalg.norm(forward), 1e-12)
    side = np.cross(forward, up)
    side /= max(np.linalg.norm(side), 1e-12)
    true_up = np.cross(side, forward)
    mat = np.eye(4, dtype=np.float64)
    mat[0, :3] = side
    mat[1, :3] = true_up
    mat[2, :3] = -forward
    mat[0, 3] = -np.dot(side, eye)
    mat[1, 3] = -np.dot(true_up, eye)
    mat[2, 3] = np.dot(forward, eye)
    return mat


# -----------------------------------------------------------------------------
# Mesh construction
# -----------------------------------------------------------------------------
def vertex_record(position: Iterable[float], normal: Iterable[float],
                  color: Iterable[float], uv: Iterable[float] = (0.0, 0.0)) -> list[float]:
    return [*position, *normal, *color, *uv]


def build_bh_surface(params: ModelParameters, center: np.ndarray) -> np.ndarray:
    m = params.black_hole_mass
    rs = params.schwarzschild_radius
    r_values = np.linspace(rs * 1.035, BH_R_MAX_FACTOR * m, 48)
    phi_values = np.linspace(0.0, 2.0 * math.pi, 144, endpoint=False)
    verts: list[list[float]] = []
    for ri in range(len(r_values) - 1):
        for pi in range(len(phi_values)):
            r0, r1 = float(r_values[ri]), float(r_values[ri + 1])
            p0 = float(phi_values[pi])
            p1 = float(phi_values[(pi + 1) % len(phi_values)])
            def sample(r: float, phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                z = 2.0 * math.sqrt(max(0.0, rs * (r - rs)))
                dz = math.sqrt(rs / max(r - rs, 1e-8))
                pos = center + np.array([r * math.cos(phi), r * math.sin(phi), z])
                n = np.array([-dz * math.cos(phi), -dz * math.sin(phi), 1.0])
                n /= np.linalg.norm(n)
                t = (r - rs) / (BH_R_MAX_FACTOR * m - rs)
                col = np.array([0.045 + 0.05 * t, 0.19 + 0.23 * t, 0.34 + 0.32 * t])
                return pos, n, col
            a = sample(r0, p0); b = sample(r1, p0)
            c = sample(r1, p1); d = sample(r0, p1)
            uv_a = ((r0 - rs) / (BH_R_MAX_FACTOR*m-rs), p0 / (2*math.pi))
            uv_b = ((r1 - rs) / (BH_R_MAX_FACTOR*m-rs), p0 / (2*math.pi))
            uv_c = ((r1 - rs) / (BH_R_MAX_FACTOR*m-rs), p1 / (2*math.pi))
            uv_d = ((r0 - rs) / (BH_R_MAX_FACTOR*m-rs), p1 / (2*math.pi))
            for s, uv in [(a,uv_a),(b,uv_b),(c,uv_c),(a,uv_a),(c,uv_c),(d,uv_d)]:
                verts.append(vertex_record(s[0], s[1], s[2], uv))
    return np.asarray(verts, dtype=np.float32)


def build_ellis_surface(params: ModelParameters, center: np.ndarray) -> np.ndarray:
    a = params.wormhole_throat
    l_values = np.linspace(-WH_L_MAX_FACTOR*a, WH_L_MAX_FACTOR*a, 72)
    phi_values = np.linspace(0.0, 2.0*math.pi, 144, endpoint=False)
    verts: list[list[float]] = []
    for li in range(len(l_values)-1):
        for pi in range(len(phi_values)):
            l0, l1 = float(l_values[li]), float(l_values[li+1])
            p0 = float(phi_values[pi])
            p1 = float(phi_values[(pi+1) % len(phi_values)])
            def sample(l: float, phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                radius = math.sqrt(l*l+a*a)
                z = a * math.asinh(l/a)
                pos = center + np.array([radius*math.cos(phi), radius*math.sin(phi), z])
                normal = np.array([a*math.cos(phi)/radius, a*math.sin(phi)/radius, -l/radius])
                normal /= np.linalg.norm(normal)
                side = 0.5 + 0.5 * math.tanh(l/(1.5*a))
                col = np.array([0.055 + 0.10*side, 0.30 + 0.17*(1-side), 0.42 + 0.32*side])
                return pos, normal, col
            a0=sample(l0,p0); b0=sample(l1,p0); c0=sample(l1,p1); d0=sample(l0,p1)
            u0=(l0/(2*WH_L_MAX_FACTOR*a)+0.5,p0/(2*math.pi))
            u1=(l1/(2*WH_L_MAX_FACTOR*a)+0.5,p0/(2*math.pi))
            u2=(l1/(2*WH_L_MAX_FACTOR*a)+0.5,p1/(2*math.pi))
            u3=(l0/(2*WH_L_MAX_FACTOR*a)+0.5,p1/(2*math.pi))
            for s, uv in [(a0,u0),(b0,u1),(c0,u2),(a0,u0),(c0,u2),(d0,u3)]:
                verts.append(vertex_record(s[0],s[1],s[2],uv))
    return np.asarray(verts, dtype=np.float32)


def build_bh_wire(params: ModelParameters, center: np.ndarray) -> np.ndarray:
    m=params.black_hole_mass; rs=params.schwarzschild_radius
    segments: list[np.ndarray] = []
    angles=np.linspace(0,2*math.pi,144,endpoint=False)
    radii=np.linspace(rs*1.05,BH_R_MAX_FACTOR*m,30)
    for r in radii[::3]:
        pts=schwarzschild_embedding(np.full_like(angles,r),angles,m)+center
        segments.append(np.stack([pts,np.roll(pts,-1,axis=0)],axis=1).reshape(-1,3))
    for phi in np.linspace(0,2*math.pi,28,endpoint=False):
        pts=schwarzschild_embedding(radii,np.full_like(radii,phi),m)+center
        segments.append(np.stack([pts[:-1],pts[1:]],axis=1).reshape(-1,3))
    return np.concatenate(segments).astype(np.float32)


def build_ellis_wire(params: ModelParameters, center: np.ndarray) -> np.ndarray:
    a=params.wormhole_throat
    angles=np.linspace(0,2*math.pi,144,endpoint=False)
    ls=np.linspace(-WH_L_MAX_FACTOR*a,WH_L_MAX_FACTOR*a,72)
    segments: list[np.ndarray]=[]
    for l in ls[::4]:
        pts=ellis_embedding(np.full_like(angles,l),angles,a)+center
        segments.append(np.stack([pts,np.roll(pts,-1,axis=0)],axis=1).reshape(-1,3))
    for phi in np.linspace(0,2*math.pi,28,endpoint=False):
        pts=ellis_embedding(ls,np.full_like(ls,phi),a)+center
        segments.append(np.stack([pts[:-1],pts[1:]],axis=1).reshape(-1,3))
    return np.concatenate(segments).astype(np.float32)


def build_sphere(center: np.ndarray, radius: float,
                 color: tuple[float,float,float], rings: int=28, sectors: int=56) -> np.ndarray:
    verts=[]
    for i in range(rings):
        lat0=-math.pi/2+math.pi*i/rings
        lat1=-math.pi/2+math.pi*(i+1)/rings
        for j in range(sectors):
            lon0=2*math.pi*j/sectors
            lon1=2*math.pi*(j+1)/sectors
            def v(lat,lon):
                normal=np.array([math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)])
                return center+radius*normal,normal
            p00,n00=v(lat0,lon0); p01,n01=v(lat0,lon1)
            p10,n10=v(lat1,lon0); p11,n11=v(lat1,lon1)
            for p,n in [(p00,n00),(p10,n10),(p11,n11),(p00,n00),(p11,n11),(p01,n01)]:
                verts.append(vertex_record(p,n,color))
    return np.asarray(verts,dtype=np.float32)


def build_accretion_disk(center: np.ndarray, mass: float,
                         segments: int=240) -> np.ndarray:
    """Art-directed luminous disk mesh; not a hydrodynamic disk solution."""
    r_inner=3.2*mass; r_outer=8.6*mass
    radii=np.linspace(r_inner,r_outer,18)
    angles=np.linspace(0,2*math.pi,segments,endpoint=False)
    z=4.3*mass
    verts=[]
    for ri in range(len(radii)-1):
        for ai in range(len(angles)):
            r0,r1=float(radii[ri]),float(radii[ri+1])
            p0=float(angles[ai]); p1=float(angles[(ai+1)%len(angles)])
            def sample(r,phi):
                pos=center+np.array([r*math.cos(phi),r*math.sin(phi),z])
                t=(r-r_inner)/(r_outer-r_inner)
                inner=np.array([1.00,0.82,0.46]); mid=np.array([1.00,0.34,0.075]); outer=np.array([0.42,0.075,0.23])
                col=(1-t)*inner+t*mid if t<0.58 else (1-(t-0.58)/0.42)*mid+((t-0.58)/0.42)*outer
                return pos,col
            a0,c0=sample(r0,p0); b0,c1=sample(r1,p0); c2p,c2=sample(r1,p1); d0,c3=sample(r0,p1)
            uv0=((r0-r_inner)/(r_outer-r_inner),p0/(2*math.pi))
            uv1=((r1-r_inner)/(r_outer-r_inner),p0/(2*math.pi))
            uv2=((r1-r_inner)/(r_outer-r_inner),p1/(2*math.pi))
            uv3=((r0-r_inner)/(r_outer-r_inner),p1/(2*math.pi))
            n=(0.0,0.0,1.0)
            for p,c,uv in [(a0,c0,uv0),(b0,c1,uv1),(c2p,c2,uv2),(a0,c0,uv0),(c2p,c2,uv2),(d0,c3,uv3)]:
                verts.append(vertex_record(p,n,c,uv))
    return np.asarray(verts,dtype=np.float32)


def build_ring_points_bh(radius: float, params: ModelParameters,
                         center: np.ndarray, z_offset: float=0.0,
                         n: int=360) -> np.ndarray:
    phi=np.linspace(0,2*math.pi,n,endpoint=True)
    pts=schwarzschild_embedding(np.full_like(phi,radius),phi,params.black_hole_mass)+center
    pts[:,2]+=z_offset
    return pts.astype(np.float32)


def build_ring_points_wh(params: ModelParameters, center: np.ndarray,
                         n: int=360) -> np.ndarray:
    phi=np.linspace(0,2*math.pi,n,endpoint=True)
    pts=ellis_embedding(np.zeros_like(phi),phi,params.wormhole_throat)+center
    return pts.astype(np.float32)


def mesh_data(points: np.ndarray, color: tuple[float,float,float],
              normal: tuple[float,float,float]=(0,0,1)) -> np.ndarray:
    pts=np.asarray(points,dtype=np.float32).reshape(-1,3)
    rows=np.zeros((len(pts),11),dtype=np.float32)
    rows[:,:3]=pts
    rows[:,3:6]=np.asarray(normal,dtype=np.float32)
    rows[:,6:9]=np.asarray(color,dtype=np.float32)
    return rows


def create_render_item(ctx: moderngl.Context, program: moderngl.Program,
                       data: np.ndarray, mode: int, group: str,
                       emission: float=0.0, style: int=0) -> RenderItem:
    arr=np.asarray(data,dtype=np.float32)
    vbo=ctx.buffer(arr.tobytes())
    vao=ctx.vertex_array(program,[(vbo,"3f 3f 3f 2f","in_position","in_normal","in_color","in_uv")])
    return RenderItem(vao,len(arr),mode,group,emission,style)


def build_scene(ctx: moderngl.Context, program: moderngl.Program,
                params: ModelParameters) -> SceneResources:
    items: list[RenderItem]=[]
    ray_items: list[RayRenderItem]=[]
    LINES=moderngl.LINES; STRIP=moderngl.LINE_STRIP; TRIANGLES=moderngl.TRIANGLES

    # Filled spatial embeddings + fine luminous coordinate wireframes.
    items.append(create_render_item(ctx,program,build_bh_surface(params,BH_CENTER),TRIANGLES,"bh",0.13))
    items.append(create_render_item(ctx,program,mesh_data(build_bh_wire(params,BH_CENTER),(0.20,0.53,0.82)),LINES,"bh",0.38))
    items.append(create_render_item(ctx,program,build_ellis_surface(params,WH_CENTER),TRIANGLES,"wh",0.15))
    items.append(create_render_item(ctx,program,mesh_data(build_ellis_wire(params,WH_CENTER),(0.16,0.78,0.86)),LINES,"wh",0.42))

    # Black-hole shadow/horizon visual and illustrative accretion disk.
    rs=params.schwarzschild_radius
    horizon_center=BH_CENTER+np.array([0.0,0.0,0.55*params.black_hole_mass])
    items.append(create_render_item(ctx,program,
        build_sphere(horizon_center,0.91*rs,(0.004,0.006,0.014)),TRIANGLES,"bh",0.05))
    items.append(create_render_item(ctx,program,
        build_accretion_disk(BH_CENTER,params.black_hole_mass),TRIANGLES,"bh",0.72,1))
    photon=build_ring_points_bh(params.photon_sphere_radius,params,BH_CENTER)
    items.append(create_render_item(ctx,program,mesh_data(photon,GOLD),STRIP,"bh",1.35))
    horizon_ring=build_ring_points_bh(rs*1.045,params,BH_CENTER)
    items.append(create_render_item(ctx,program,mesh_data(horizon_ring,(1.0,0.30,0.17)),STRIP,"bh",0.95))
    # Additional inner/outer disk guide rings.
    for radius,col in [(3.2*params.black_hole_mass,(1.0,0.88,0.60)),
                       (5.4*params.black_hole_mass,(1.0,0.38,0.10)),
                       (8.5*params.black_hole_mass,(0.58,0.14,0.30))]:
        ring=build_ring_points_bh(radius,params,BH_CENTER,z_offset=4.3*params.black_hole_mass)
        items.append(create_render_item(ctx,program,mesh_data(ring,col),STRIP,"bh",0.75))

    # Ellis throat ring, both asymptotic regions and optional guide circles.
    throat=build_ring_points_wh(params,WH_CENTER)
    items.append(create_render_item(ctx,program,mesh_data(throat,CYAN),STRIP,"wh",1.55))
    for side in (-1.0,1.0):
        l=side*2.0*params.wormhole_throat
        ring_pts=ellis_embedding(np.full(240,l),np.linspace(0,2*math.pi,240),params.wormhole_throat)+WH_CENTER
        col=(0.31,0.52,1.0) if side<0 else (0.70,0.40,1.0)
        items.append(create_render_item(ctx,program,mesh_data(ring_pts,col),STRIP,"wh",0.55))

    # Exact-background null geodesics from the physics module.
    rays=build_geodesics(params,BH_CENTER,WH_CENTER)
    for ray in rays:
        data=mesh_data(ray.points_3d,ray.color)
        item=create_render_item(ctx,program,data,STRIP,ray.model,0.48)
        ray_items.append(RayRenderItem(ray,item))

    # Marker buffer is dynamically updated with the current point on each ray.
    capacity=max(1,len(rays))
    marker_buffer=ctx.buffer(reserve=capacity*11*4)
    marker_vao=ctx.vertex_array(program,[(marker_buffer,"3f 3f 3f 2f","in_position","in_normal","in_color","in_uv")])
    return SceneResources(items,ray_items,marker_buffer,marker_vao,capacity)


# -----------------------------------------------------------------------------
# Procedural sky and post-processing targets
# -----------------------------------------------------------------------------
def create_fullscreen_quad(ctx: moderngl.Context) -> moderngl.Buffer:
    data=np.array([
        [-1.0,-1.0,0.0,0.0], [1.0,-1.0,1.0,0.0],
        [-1.0,1.0,0.0,1.0], [1.0,1.0,1.0,1.0],
    ],dtype=np.float32)
    buf=ctx.buffer(data.tobytes())
    # VAO depends on program; the caller creates a VAO per program.
    return buf


class RenderTargets:
    def __init__(self,ctx: moderngl.Context) -> None:
        self.ctx=ctx
        self.size=(0,0)
        self.scene_tex=None; self.depth=None; self.scene_fbo=None
        self.bright_tex=None; self.bright_fbo=None
        self.blur_tex=None; self.blur_fbo=None

    def resize(self,screen_w:int,screen_h:int,scale:float) -> bool:
        iw=max(480,int(screen_w*scale)); ih=max(270,int(screen_h*scale))
        # Align dimensions to multiples of 8 for predictable raster allocation.
        iw=max(8,(iw//8)*8); ih=max(8,(ih//8)*8)
        if (iw,ih)==self.size:
            return False
        self.release()
        self.size=(iw,ih)
        self.scene_tex=self.ctx.texture((iw,ih),4,dtype="f2")
        self.scene_tex.filter=(moderngl.LINEAR,moderngl.LINEAR)
        self.depth=self.ctx.depth_renderbuffer((iw,ih))
        self.scene_fbo=self.ctx.framebuffer([self.scene_tex],self.depth)
        bw,bh=max(1,iw//2),max(1,ih//2)
        self.bright_tex=self.ctx.texture((bw,bh),4,dtype="f2")
        self.blur_tex=self.ctx.texture((bw,bh),4,dtype="f2")
        self.bright_tex.filter=(moderngl.LINEAR,moderngl.LINEAR)
        self.blur_tex.filter=(moderngl.LINEAR,moderngl.LINEAR)
        self.bright_fbo=self.ctx.framebuffer([self.bright_tex])
        self.blur_fbo=self.ctx.framebuffer([self.blur_tex])
        return True

    def release(self) -> None:
        for obj in [self.scene_fbo,self.bright_fbo,self.blur_fbo,
                    self.scene_tex,self.depth,self.bright_tex,self.blur_tex]:
            if obj is not None:
                obj.release()
        self.scene_tex=self.depth=self.scene_fbo=None
        self.bright_tex=self.bright_fbo=self.blur_tex=self.blur_fbo=None
        self.size=(0,0)


class DynamicResolution:
    def __init__(self) -> None:
        self.scale=0.90
        self.samples: deque[float]=deque(maxlen=48)
        self.last_adjust=time.perf_counter()
        self.enabled=True
        self.last_change="initial scale"

    def update(self,frame_ms:float) -> bool:
        self.samples.append(frame_ms)
        if not self.enabled or len(self.samples)<24:
            return False
        now=time.perf_counter()
        if now-self.last_adjust<1.0:
            return False
        self.last_adjust=now
        avg=sum(self.samples)/len(self.samples)
        old=self.scale
        if avg>TARGET_FRAME_MS*1.12:
            self.scale=max(MIN_RENDER_SCALE,self.scale-0.055)
            self.last_change=f"GPU/CPU frame budget high ({avg:.1f} ms)"
        elif avg<TARGET_FRAME_MS*0.82:
            self.scale=min(MAX_RENDER_SCALE,self.scale+0.025)
            self.last_change=f"headroom available ({avg:.1f} ms)"
        return abs(old-self.scale)>1e-4


# -----------------------------------------------------------------------------
# Overlay / UI
# -----------------------------------------------------------------------------
class HUD:
    def __init__(self) -> None:
        pygame.font.init()
        self.font_cache: dict[int,pygame.font.Font]={}

    def font(self,size:int) -> pygame.font.Font:
        size=max(12,int(size))
        if size not in self.font_cache:
            self.font_cache[size]=pygame.font.SysFont("segoeui",size)
        return self.font_cache[size]

    def text(self,surface:pygame.Surface,text:str,x:int,y:int,size:int=17,
             color=(220,232,255),bold:bool=False) -> None:
        f=self.font(size)
        if bold:
            f.set_bold(True)
        surface.blit(f.render(text,True,color),(x,y))
        if bold:
            f.set_bold(False)

    def panel(self,surface:pygame.Surface,rect:pygame.Rect,
              fill=(7,13,30,205),border=(72,116,166,95),radius=14) -> None:
        pygame.draw.rect(surface,fill,rect,border_radius=radius)
        pygame.draw.rect(surface,border,rect,1,border_radius=radius)

    def draw(self,size:tuple[int,int],mode:str,scene_mode:str,params:ModelParameters,
             paused:bool,speed:float,fps:float,scale:float,renderer:str,
             elapsed:float,show_hud:bool,ray_count:int) -> pygame.Surface:
        w,h=size
        surf=pygame.Surface((w,h),pygame.SRCALPHA,32)
        if not show_hud:
            return surf
        k=max(0.85,min(1.5,h/900.0))
        pad=int(22*k)
        # Brand panel.
        brand=pygame.Rect(pad,pad,int(410*k),int(94*k))
        self.panel(surf,brand)
        self.text(surf,"EVENT HORIZON",brand.x+18,brand.y+12,int(24*k),(235,244,255),True)
        self.text(surf,"CINEMATIC RELATIVITY OBSERVATORY",brand.x+19,brand.y+47,int(12*k),(119,204,235))
        status="PAUSED" if paused else "LIVE"
        self.text(surf,f"{status}  /  {scene_mode.upper()}  /  {mode.upper()} MODE",
                  brand.x+19,brand.y+66,int(11*k),(255,190,104) if paused else (116,226,190),True)

        # Top-right runtime panel.
        runtime=pygame.Rect(w-int(325*k)-pad,pad,int(325*k),int(94*k))
        self.panel(surf,runtime)
        self.text(surf,f"{fps:5.1f} FPS",runtime.x+16,runtime.y+12,int(21*k),(226,240,255),True)
        self.text(surf,f"INTERNAL RENDER  {scale*100:3.0f}%",runtime.x+16,runtime.y+42,int(12*k),(112,206,238))
        self.text(surf,renderer[:34],runtime.x+16,runtime.y+65,int(10*k),(160,174,205))

        if mode=="lab":
            left=pygame.Rect(pad,int(140*k),int(335*k),int(300*k))
            self.panel(surf,left)
            self.text(surf,"LABORATORY",left.x+18,left.y+15,int(16*k),(235,244,255),True)
            self.text(surf,"FIXED-BACKGROUND METRICS",left.x+18,left.y+43,int(10*k),(111,192,224),True)
            m=params.black_hole_mass; a=params.wormhole_throat
            lines=[
                ("SCHWARZSCHILD",(255,177,107)),
                (f"Mass M                 {m:6.3f}",(220,230,248)),
                (f"Horizon r_s            {params.schwarzschild_radius:6.3f}",(220,230,248)),
                (f"Photon sphere r_ph     {params.photon_sphere_radius:6.3f}",(220,230,248)),
                (f"Critical b             {params.critical_impact_parameter:6.3f}",(220,230,248)),
                ("ELLIS-BRONNIKOV",(117,225,244)),
                (f"Throat radius a        {a:6.3f}",(220,230,248)),
                (f"Null ray paths         {ray_count:6d}",(220,230,248)),
            ]
            yy=left.y+68
            for line,col in lines:
                self.text(surf,line,left.x+18,yy,int(12*k),col,line.isupper())
                yy+=int(26*k)
            self.text(surf,"[ / ]  adjust M     , / .  adjust a",left.x+18,left.bottom-35,int(10*k),(255,202,133))

            right=pygame.Rect(w-int(355*k)-pad,int(140*k),int(355*k),int(300*k))
            self.panel(surf,right)
            self.text(surf,"MODEL NOTES",right.x+18,right.y+15,int(16*k),(235,244,255),True)
            notes=[
                "BH: vacuum Schwarzschild background",
                "WH: Ellis metric / phantom scalar source",
                "Rays: equatorial null geodesics",
                "Solver: CPU RK4, fixed metrics",
                "Renderer: OpenGL + procedural shaders",
                "Embedding height is not a real dimension.",
                "No BH-WH collision metric is evolved.",
            ]
            yy=right.y+55
            for line in notes:
                self.text(surf,line,right.x+18,yy,int(11*k),(188,205,229))
                yy+=int(29*k)

        bottom=pygame.Rect(pad,h-int(74*k)-pad,w-2*pad,int(74*k))
        self.panel(surf,bottom,fill=(6,11,25,182),border=(72,116,166,75),radius=12)
        if mode=="explore":
            line1="DRAG  orbit     RIGHT-DRAG  pan     WHEEL  zoom     1  black hole     2  wormhole     3  both"
            line2="TAB  laboratory / explore     SPACE  pause     UP/DOWN  animation speed     H  hide HUD     ESC  quit"
        else:
            line1="LAB:  [ / ]  change BH mass     , / .  change wormhole throat     1 / 2 / 3  select geometry"
            line2="DRAG  orbit     RIGHT-DRAG  pan     WHEEL  zoom     TAB  explore     SPACE  pause     R  reset"
        self.text(surf,line1,bottom.x+18,bottom.y+12,int(11*k),(206,221,245))
        self.text(surf,line2,bottom.x+18,bottom.y+39,int(11*k),(121,191,223))
        return surf


# -----------------------------------------------------------------------------
# Utility and app
# -----------------------------------------------------------------------------
def make_quad_vao(ctx:moderngl.Context,program:moderngl.Program,buffer:moderngl.Buffer) -> moderngl.VertexArray:
    return ctx.vertex_array(program,[(buffer,"2f 2f","in_position","in_uv")])


def release_programs(objects:list) -> None:
    for obj in objects:
        try: obj.release()
        except Exception: pass


def create_overlay_texture(ctx:moderngl.Context,size:tuple[int,int]) -> moderngl.Texture:
    tex=ctx.texture(size,4,dtype="f1")
    tex.filter=(moderngl.LINEAR,moderngl.LINEAR)
    return tex


def main() -> int:
    params=ModelParameters()
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION,3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION,3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER,1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE,24)
    flags=pygame.OPENGL|pygame.DOUBLEBUF|pygame.RESIZABLE
    try:
        display_info=pygame.display.Info()
        win_w=min(DEFAULT_WINDOW[0],max(960,display_info.current_w))
        win_h=min(DEFAULT_WINDOW[1],max(540,display_info.current_h))
        pygame.display.set_mode((win_w,win_h),flags,vsync=0)
        pygame.display.set_caption(APP_NAME)
        ctx=moderngl.create_context(require=330)
    except Exception as exc:
        print("OpenGL startup failed:",exc)
        print("Please update your NVIDIA/AMD graphics driver and verify OpenGL 3.3 support.")
        pygame.quit()
        return 2

    print("\n"+APP_NAME)
    print("="*len(APP_NAME))
    print("OpenGL vendor   :",ctx.info.get("GL_VENDOR","Unknown"))
    print("OpenGL renderer :",ctx.info.get("GL_RENDERER","Unknown"))
    print("OpenGL version  :",ctx.info.get("GL_VERSION","Unknown"))
    print("Physics         : CPU / NumPy / RK4 exact-background null geodesics")
    print("Post processing : adaptive resolution + bloom + filmic vignette")
    print("Scientific scope: separate fixed Schwarzschild and Ellis metrics; no collision evolution.\n")

    # Programs / shared fullscreen quad.
    mesh_program=ctx.program(vertex_shader=MESH_VERTEX_SHADER,fragment_shader=MESH_FRAGMENT_SHADER)
    bg_program=ctx.program(vertex_shader=QUAD_VERTEX_SHADER,fragment_shader=BACKGROUND_FRAGMENT_SHADER)
    bright_program=ctx.program(vertex_shader=QUAD_VERTEX_SHADER,fragment_shader=BRIGHT_FRAGMENT_SHADER)
    blur_program=ctx.program(vertex_shader=QUAD_VERTEX_SHADER,fragment_shader=BLUR_FRAGMENT_SHADER)
    final_program=ctx.program(vertex_shader=QUAD_VERTEX_SHADER,fragment_shader=FINAL_FRAGMENT_SHADER)
    overlay_program=ctx.program(vertex_shader=QUAD_VERTEX_SHADER,fragment_shader=OVERLAY_FRAGMENT_SHADER)
    quad_buffer=create_fullscreen_quad(ctx)
    bg_vao=make_quad_vao(ctx,bg_program,quad_buffer)
    bright_vao=make_quad_vao(ctx,bright_program,quad_buffer)
    blur_vao=make_quad_vao(ctx,blur_program,quad_buffer)
    final_vao=make_quad_vao(ctx,final_program,quad_buffer)
    overlay_vao=make_quad_vao(ctx,overlay_program,quad_buffer)

    targets=RenderTargets(ctx)
    dynamic=DynamicResolution()
    hud=HUD()
    resources=build_scene(ctx,mesh_program,params)

    ctx.enable(moderngl.DEPTH_TEST|moderngl.PROGRAM_POINT_SIZE)
    ctx.depth_func="<"
    ctx.disable(moderngl.CULL_FACE)
    ctx.line_width=1.0

    overlay_texture=None
    overlay_size=(0,0)
    hud_refresh_timer=1.0
    clock=pygame.time.Clock()
    running=True
    paused=False
    hide_hud=False
    animation_speed=1.0
    elapsed=0.0
    scene_mode="both"       # bh / wh / both
    interaction_mode="explore"  # explore / lab
    yaw=math.radians(-33.0)
    elevation=math.radians(24.0)
    distance=67.0
    target=np.array([0.0,0.0,2.0],dtype=np.float64)
    dragging=False
    panning=False
    last_mouse=(0,0)
    frame_ms=16.7
    fps_smooth=60.0

    def rebuild_scene() -> None:
        nonlocal resources
        resources.release()
        resources=build_scene(ctx,mesh_program,params)

    def focus(which:str) -> None:
        nonlocal scene_mode,target,distance
        scene_mode=which
        if which=="bh":
            target=BH_CENTER+np.array([0.0,0.0,3.0*params.black_hole_mass])
            distance=28.0*params.black_hole_mass
        elif which=="wh":
            target=WH_CENTER.copy()
            distance=16.0*params.wormhole_throat
        else:
            target=np.array([0.0,0.0,2.0],dtype=np.float64)
            distance=67.0

    try:
        while running:
            frame_start=time.perf_counter()
            raw_dt=clock.tick(0)/1000.0
            dt=min(raw_dt,0.05)
            fps=1.0/max(raw_dt,1e-6)
            fps_smooth=0.90*fps_smooth+0.10*fps
            w,h=pygame.display.get_window_size()
            w=max(640,w); h=max(360,h)

            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running=False
                elif event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_ESCAPE:
                        running=False
                    elif event.key==pygame.K_SPACE:
                        paused=not paused
                    elif event.key==pygame.K_TAB:
                        interaction_mode="lab" if interaction_mode=="explore" else "explore"
                    elif event.key==pygame.K_e:
                        interaction_mode="explore"
                    elif event.key==pygame.K_l:
                        interaction_mode="lab"
                    elif event.key==pygame.K_h:
                        hide_hud=not hide_hud
                    elif event.key==pygame.K_UP:
                        animation_speed=min(6.0,animation_speed*1.25)
                    elif event.key==pygame.K_DOWN:
                        animation_speed=max(0.10,animation_speed/1.25)
                    elif event.key==pygame.K_r:
                        elapsed=0.0
                    elif event.key==pygame.K_1:
                        focus("bh")
                    elif event.key==pygame.K_2:
                        focus("wh")
                    elif event.key==pygame.K_3:
                        focus("both")
                    elif interaction_mode=="lab" and event.key==pygame.K_LEFTBRACKET:
                        params.black_hole_mass=max(0.55,params.black_hole_mass-0.05)
                        rebuild_scene()
                    elif interaction_mode=="lab" and event.key==pygame.K_RIGHTBRACKET:
                        params.black_hole_mass=min(2.0,params.black_hole_mass+0.05)
                        rebuild_scene()
                    elif interaction_mode=="lab" and event.key==pygame.K_COMMA:
                        params.wormhole_throat=max(0.65,params.wormhole_throat-0.10)
                        rebuild_scene()
                    elif interaction_mode=="lab" and event.key==pygame.K_PERIOD:
                        params.wormhole_throat=min(4.0,params.wormhole_throat+0.10)
                        rebuild_scene()
                elif event.type==pygame.MOUSEBUTTONDOWN:
                    if event.button==1:
                        dragging=True; last_mouse=event.pos
                    elif event.button==3:
                        panning=True; last_mouse=event.pos
                elif event.type==pygame.MOUSEBUTTONUP:
                    if event.button==1: dragging=False
                    elif event.button==3: panning=False
                elif event.type==pygame.MOUSEMOTION:
                    if dragging:
                        dx=event.pos[0]-last_mouse[0]; dy=event.pos[1]-last_mouse[1]
                        yaw+=dx*0.0055
                        elevation=max(-1.40,min(1.40,elevation-dy*0.0055))
                        last_mouse=event.pos
                    elif panning:
                        dx=event.pos[0]-last_mouse[0]; dy=event.pos[1]-last_mouse[1]
                        pan_scale=distance*0.0018
                        right=np.array([math.cos(yaw),math.sin(yaw),0.0])
                        up=np.array([0.0,0.0,1.0])
                        target += -right*dx*pan_scale + up*dy*pan_scale
                        last_mouse=event.pos
                elif event.type==pygame.MOUSEWHEEL:
                    distance*=math.exp(-event.y*0.12)
                    distance=max(7.0,min(180.0,distance))

            if not paused:
                elapsed+=dt*animation_speed
            hud_refresh_timer += raw_dt

            # Adaptive internal resolution based on recent end-to-end frame time.
            if targets.resize(w,h,dynamic.scale):
                pass
            elif targets.size==(0,0):
                targets.resize(w,h,dynamic.scale)
            # If screen size changed, RenderTargets.resize catches it. The
            # overlay uses actual display resolution and is recreated as needed.

            horizontal=distance*math.cos(elevation)
            eye=target+np.array([
                horizontal*math.sin(yaw),
                -horizontal*math.cos(yaw),
                distance*math.sin(elevation),
            ])
            iw,ih=targets.size
            targets.scene_fbo.use()
            ctx.viewport=(0,0,iw,ih)
            targets.scene_fbo.clear(0.0,0.0,0.0,1.0,depth=1.0)

            # Procedural nebula / starfield background, camera-parallax tinted.
            ctx.disable(moderngl.DEPTH_TEST)
            bg_program["u_resolution"].value=(float(iw),float(ih))
            bg_program["u_time"].value=elapsed
            bg_program["u_camera_offset"].value=(yaw,elevation)
            bg_vao.render(mode=moderngl.TRIANGLE_STRIP,vertices=4)
            ctx.enable(moderngl.DEPTH_TEST)

            proj=perspective(48.0,iw/ih,0.1,500.0)
            view=look_at(eye,target,np.array([0.0,0.0,1.0]))
            mvp=(proj@view).astype("f4")
            mesh_program["u_mvp"].write(mvp.T.tobytes())
            mesh_program["u_camera"].value=tuple(float(x) for x in eye)
            mesh_program["u_time"].value=elapsed
            mesh_program["u_point_size"].value=12.0
            mesh_program["u_draw_point"].value=0

            for item in resources.items:
                if scene_mode!="both" and item.group!=scene_mode:
                    continue
                mesh_program["u_emission"].value=item.emission
                mesh_program["u_style"].value=item.style
                item.vao.render(mode=item.mode,vertices=item.count)

            for ray in resources.ray_items:
                if scene_mode!="both" and ray.geodesic.model!=scene_mode:
                    continue
                mesh_program["u_emission"].value=0.52
                mesh_program["u_style"].value=0
                ray.item.vao.render(mode=moderngl.LINE_STRIP,vertices=ray.item.count)

            # Moving photon markers: one luminous point per geodesic.
            marker_rows=[]
            for ray in resources.ray_items:
                if scene_mode!="both" and ray.geodesic.model!=scene_mode:
                    continue
                pts=ray.geodesic.points_3d
                if len(pts)==0: continue
                index=int((elapsed*25.0) % len(pts))
                p=pts[index]
                marker_rows.append([*p,0.0,0.0,1.0,*ray.geodesic.color,0.0,0.0])
            if marker_rows:
                data=np.asarray(marker_rows,dtype=np.float32)
                resources.marker_buffer.orphan(data.nbytes)
                resources.marker_buffer.write(data.tobytes())
                mesh_program["u_emission"].value=1.65
                mesh_program["u_style"].value=0
                mesh_program["u_draw_point"].value=1
                mesh_program["u_point_size"].value=max(7.0,min(15.0,h/78.0))
                resources.marker_vao.render(mode=moderngl.POINTS,vertices=len(marker_rows))
                mesh_program["u_draw_point"].value=0

            # Bright extraction and two-axis bloom blur.
            targets.bright_fbo.use()
            bw,bh=targets.bright_tex.size
            ctx.viewport=(0,0,bw,bh)
            ctx.disable(moderngl.DEPTH_TEST)
            targets.scene_tex.use(location=0)
            bright_program["u_scene"].value=0
            bright_vao.render(mode=moderngl.TRIANGLE_STRIP,vertices=4)

            targets.blur_fbo.use()
            targets.bright_tex.use(location=0)
            blur_program["u_image"].value=0
            blur_program["u_direction"].value=(1.0,0.0)
            blur_program["u_texel"].value=(1.0/bw,1.0/bh)
            blur_vao.render(mode=moderngl.TRIANGLE_STRIP,vertices=4)

            targets.bright_fbo.use()
            targets.blur_tex.use(location=0)
            blur_program["u_image"].value=0
            blur_program["u_direction"].value=(0.0,1.0)
            blur_program["u_texel"].value=(1.0/bw,1.0/bh)
            blur_vao.render(mode=moderngl.TRIANGLE_STRIP,vertices=4)

            # Final tone mapping / upscaling on the display framebuffer.
            ctx.screen.use()
            ctx.viewport=(0,0,w,h)
            ctx.clear(0.0,0.0,0.0,1.0)
            targets.scene_tex.use(location=0)
            targets.bright_tex.use(location=1)
            final_program["u_scene"].value=0
            final_program["u_bloom"].value=1
            final_program["u_bloom_strength"].value=0.82
            final_program["u_time"].value=elapsed
            final_vao.render(mode=moderngl.TRIANGLE_STRIP,vertices=4)

            # Pygame-rendered crisp text/controls are uploaded as a transparent GL layer.
            if overlay_texture is None or overlay_size!=(w,h):
                if overlay_texture is not None: overlay_texture.release()
                overlay_size=(w,h)
                overlay_texture=create_overlay_texture(ctx,overlay_size)
                hud_refresh_timer=1.0
            if hud_refresh_timer >= 0.15:
                model_name={"bh":"SCHWARZSCHILD","wh":"ELLIS WORMHOLE","both":"DUAL GEOMETRY"}[scene_mode]
                renderer=str(ctx.info.get("GL_RENDERER","OpenGL renderer"))
                overlay=hud.draw((w,h),interaction_mode,model_name,params,paused,
                                 animation_speed,fps_smooth,dynamic.scale,renderer,
                                 elapsed,not hide_hud,len(resources.ray_items))
                overlay_texture.write(pygame.image.tostring(overlay,"RGBA",True))
                hud_refresh_timer=0.0
            overlay_texture.use(location=0)
            overlay_program["u_overlay"].value=0
            ctx.enable(moderngl.BLEND)
            ctx.blend_func=(moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
            overlay_vao.render(mode=moderngl.TRIANGLE_STRIP,vertices=4)
            ctx.disable(moderngl.BLEND)
            pygame.display.flip()

            frame_ms=(time.perf_counter()-frame_start)*1000.0
            if dynamic.update(frame_ms):
                # Resize/recreate at the newly selected internal render scale.
                targets.resize(w,h,dynamic.scale)

    except KeyboardInterrupt:
        pass
    finally:
        resources.release()
        targets.release()
        if overlay_texture is not None: overlay_texture.release()
        release_programs([bg_vao,bright_vao,blur_vao,final_vao,overlay_vao,quad_buffer,
                          mesh_program,bg_program,bright_program,blur_program,final_program,overlay_program])
        ctx.release()
        pygame.quit()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
