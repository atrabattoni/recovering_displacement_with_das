import numpy as np
import xarray as xr

# Source Time Functions


def sinus(t, f0=1.0):
    return np.sin(2 * np.pi * t * f0)


def ricker(t, f0=1.0):
    "Ricker function."
    w = 2 * np.pi * f0
    x = (w * t) ** 2 / 2
    return (1 - x) * np.exp(-x / 2)


# Sources


def plane_wave(t, r, f0=1.0, p=1.0, amp=1.0, kind="p", stf=ricker):
    if kind == "p":
        u = amp * p / np.abs(p)
    elif kind == "s":
        u = amp * 1j * p / np.abs(p)
    else:
        raise (ValueError("wave kind must be either `p` or `s`"))
    x = np.real(r * np.conj(p)) - t
    return u * stf(x, f0)


def spherical_wave(t, r, f0=1.0, p=1.0, amp=1.0, rs=0.0, kind="p", stf=ricker):
    dr = r - rs
    er = dr / np.abs(dr)
    if kind == "p":
        u = amp * er
    elif kind == "s":
        u = amp * 1j * er
    else:
        raise (ValueError("wave kind must be either `p` or `s`"))
    x = np.abs(dr) * p - t
    return u * stf(x, f0)


def get_source(type, **params):
    if type == "spheric":
        return lambda t, r: spherical_wave(t, r, **params)
    elif type == "plane":
        return lambda t, r: plane_wave(t, r, **params)


# 2D homogeneous simulation


def simulate(rp, t, source, g=None, ds=None, s=None, output="deformation"):
    sp = get_cumulative_distance(rp)
    sm, s, r = discretize(sp, rp, ds=ds, s=s)
    if g is not None:
        g = g(s)
    if output == "deformation":
        data = compute_deformation(t, r, source, g=g)
    elif output == "displacement":
        data = compute_displacement(t, r, source, g=g)
    out = xr.DataArray(data, {"time": t, "distance": s})
    out = out.sel(distance=sm)
    return out


def compute_deformation(t, r, source, g=None):
    dr = np.diff(r)
    es = dr / np.abs(dr)
    u = source(t[:, None], r[None, :])
    if g is not None:
        u *= g
    du = np.real(u[:, 1:] * np.conj(es)) - np.real(u[:, :-1] * np.conj(es))
    data = np.pad(np.cumsum(du, axis=1), ((0, 0), (1, 0)))
    return data


def compute_displacement(t, r, source, g=None):
    dr = np.diff(r)
    es = dr / np.abs(dr)
    es = np.append(es, 0.0)
    u = source(t[:, None], r[None, :])
    if g is not None:
        u *= g
    return np.real(u * np.conj(es))


def get_cumulative_distance(rp):
    sp = np.pad(np.cumsum(np.abs(np.diff(rp))), (1, 0))
    return sp


def discretize(sp, rp, ds=None, s=None):
    if ds is not None:
        n = int(sp[-1] // ds)
        L = n * ds
        sm = np.linspace(0.0, L, n + 1)
    elif s is not None:
        sm = s
    rm = np.interp(sm, sp, rp)
    s = np.concatenate((sp, sm))
    r = np.concatenate((rp, rm))
    index = np.argsort(s)
    s = s[index]
    r = r[index]
    s, index = np.unique(s, return_index=True)
    r = r[index]
    return sm, s, r
