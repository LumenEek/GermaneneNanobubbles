"""Self-contained TBPLaS model and analysis for the 2026-09-23 Germanene exports.

Lengths are nm, energies eV, and the display frame is X=-y, Y=x.
TBPLaS owns the lattice, hoppings, and sparse Hamiltonian. Bubble deformation,
PMF, site selection, and spectral postprocessing implement the MATLAB methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tbplas as tb


HBAR_EV_S = 6.582119569e-16


@dataclass(frozen=True)
class ExportParams:
    """Values saved in HBTB_cut_20260923_002028 and HBTB_roi_20260923_001017."""

    a: float = 0.402
    buckle: float = 0.06874961430369068
    hopping: float = -0.8640229585996059
    lambda_so_mev: float = 100.0
    lambda_r2_mev: float = 10.7
    radius: float = 3.0
    height: float = 1.0
    lfac: float = 5.0
    energy_min: float = -1.0
    energy_max: float = 1.0
    energy_points: int = 1201
    eta: float = 0.010
    recursion_depth: int = 1500
    roi_random_vectors: int = 8
    profile: str = "gaussian"

    @property
    def bond(self):
        return self.a / np.sqrt(3.0)

    @property
    def d0(self):
        return float(np.hypot(self.bond, self.buckle))

    @property
    def gamma0(self):
        n2 = (self.buckle / self.d0) ** 2
        cutoff = 1.0 / (1.0 + np.exp(-2.0 * self.d0 / 0.01))
        return abs(self.hopping) / ((1.0 - n2) * cutoff)

    @property
    def dimensions(self):
        side = 2.0 * self.lfac * self.radius
        nx = max(3, 3 * int(np.ceil(round(side / self.a) / 3.0)))
        ny = max(2, int(np.floor(side / (np.sqrt(3.0) * self.a) + 0.5)))
        return nx, ny, 1

    @property
    def energy(self):
        return np.linspace(self.energy_min, self.energy_max, self.energy_points)


@dataclass
class Model:
    sample: tb.Sample
    params: ExportParams
    dim: tuple[int, int, int]
    center: np.ndarray
    height: np.ndarray
    strain: np.ndarray
    display_xy: np.ndarray  # one row per physical site, not per spin orbital
    is_a: np.ndarray  # one flag per physical site
    nearest_mask: np.ndarray

    def hamiltonian(self):
        return self.sample.build_ham_csr().tocsr()


def gaussian_height(x, y, p: ExportParams):
    return p.height * np.exp(-(np.asarray(x) ** 2 + np.asarray(y) ** 2)
                              / (2.0 * (p.radius / 3.0) ** 2))


def spherical_cap_profile(x, y, p: ExportParams):
    """Exact cap height and analytic slopes, with the flat-side normal at the rim.

    Radius is the footprint radius, not the radius of curvature. The join is
    continuous in height but not slope, matching notebook 08's unrelaxed cap.
    """
    if not (np.isfinite(p.radius) and np.isfinite(p.height)
            and p.radius > 0 and 0 <= p.height < p.radius):
        raise ValueError("Spherical caps require finite 0 <= height < radius.")
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    z, hx, hy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    if p.height == 0:
        return z, hx, hy
    r2 = x*x + y*y
    inside = r2 < p.radius**2
    sphere = (p.radius**2 + p.height**2) / (2*p.height)
    root = np.sqrt(sphere**2 - r2[inside])
    z[inside] = (p.radius**2-r2[inside]) / (root+sphere-p.height)
    hx[inside], hy[inside] = -x[inside]/root, -y[inside]/root
    return z, hx, hy


def bubble_height(x, y, p: ExportParams):
    """Mid-surface height for the selected profile."""
    if p.profile == "gaussian":
        return gaussian_height(x, y, p)
    if p.profile == "spherical_cap":
        return spherical_cap_profile(x, y, p)[0]
    raise ValueError(f"Unknown bubble profile: {p.profile}")


def _rect_cell(p: ExportParams):
    a, b = p.a, p.bond
    vectors = np.array([[a, 0.0, 0.0], [0.0, 3.0 * b, 0.0], [0.0, 0.0, 1.0]])
    basis = np.array([[0.0, 3.0 * b], [0.0, b],
                      [0.5 * a, 1.5 * b], [0.5 * a, 2.5 * b]])
    sub_a = np.array([True, False, True, False])
    cell = tb.PrimitiveCell(vectors, unit=tb.NM)
    for xy, a_site in zip(basis, sub_a):
        z = (0.5 if a_site else -0.5) * p.buckle
        for spin in ("up", "down"):
            cell.add_orbital((xy[0] / a, xy[1] / (3.0 * b), z),
                             label=("A" if a_site else "B") + ":" + spin)

    def half(n1, n2, s, sp):
        return n1 > 0 or (n1 == 0 and n2 > 0) or (n1 == n2 == 0 and sp > s)

    terms = []
    for s in range(4):
        for sp in range(4):
            for n1 in range(-2, 3):
                for n2 in range(-2, 3):
                    d = np.array([n1 * a, n2 * 3.0 * b]) + basis[sp] - basis[s]
                    length = np.linalg.norm(d)
                    if length > 1e-9 and half(n1, n2, s, sp):
                        terms.append((s, sp, n1, n2, d, length))

    for s, sp, n1, n2, _, length in terms:
        if abs(length - b) < 1e-7:
            for spin in (0, 1):
                cell.add_hopping((n1, n2, 0), 2 * s + spin, 2 * sp + spin,
                                 p.hopping)

    lam = 1e-3 * p.lambda_so_mev / (3.0 * np.sqrt(3.0))
    for s, sp, n1, n2, d, length in terms:
        if abs(length - a) >= 1e-7 or sub_a[s] != sub_a[sp]:
            continue
        nu = 0.0
        for sk in range(4):
            if sub_a[sk] == sub_a[s]:
                continue
            for k1 in range(-1, 2):
                for k2 in range(-1, 2):
                    d1 = np.array([k1 * a, k2 * 3.0 * b]) + basis[sk] - basis[s]
                    d2 = d - d1
                    if abs(np.linalg.norm(d1) - b) < 1e-7 and abs(np.linalg.norm(d2) - b) < 1e-7:
                        nu = np.sign(np.cross(d1, d2))
                        break
                if nu:
                    break
            if nu:
                break
        if not nu:
            raise RuntimeError("Cannot assign Kane-Mele chirality")
        cell.add_hopping((n1, n2, 0), 2 * s, 2 * sp, 1j * nu * lam)
        cell.add_hopping((n1, n2, 0), 2 * s + 1, 2 * sp + 1, -1j * nu * lam)
        mu = 1.0 if sub_a[s] else -1.0
        dh = d / length
        lr = 1e-3 * p.lambda_r2_mev * (2.0 / 3.0) * mu
        cell.add_hopping((n1, n2, 0), 2 * s, 2 * sp + 1,
                         lr * (dh[0] - 1j * dh[1]))
        cell.add_hopping((n1, n2, 0), 2 * s + 1, 2 * sp,
                         -lr * (dh[0] + 1j * dh[1]))
    return cell, basis, sub_a


def build_model(p: ExportParams | None = None, *, flat=False):
    """Periodic spinful Germanene sample with a prescribed bubble and global pz axis."""
    p = p or ExportParams()
    nx, ny, nz = p.dimensions
    cell, basis, sub_a_basis = _rect_cell(p)
    a, b = p.a, p.bond
    center = np.array([a * np.floor(nx / 2) + 0.5 * a,
                       3.0 * b * np.floor(ny / 2) + 0.5 * b])
    primitive = np.tile(np.arange(8), nx * ny)
    ix = np.repeat(np.arange(nx), ny * 8)
    iy = np.tile(np.repeat(np.arange(ny), 8), nx)
    physical_basis = primitive // 2
    is_a = sub_a_basis[physical_basis]
    spin_up = primitive % 2 == 0
    base_x = ix * a + basis[physical_basis, 0]
    base_y = iy * 3.0 * b + basis[physical_basis, 1]
    display_x = -(base_y - center[1])
    display_y = base_x - center[0]
    height = bubble_height(display_x, display_y, p)
    if p.profile == "spherical_cap":
        # The exact cap is zero near every periodic boundary; no image overlaps.
        margin_x = min(center[0], nx*a-center[0])
        margin_y = min(center[1], ny*3*b-center[1])
        if p.radius >= min(margin_x, margin_y):
            raise ValueError("Cap footprint must lie strictly inside the periodic cell.")
    if flat:
        height = np.zeros_like(height)
    hs = 0.002  # 0.02 Angstrom, as in hb_place_atoms.m
    if flat:
        dhdx = np.zeros_like(height)
        dhdy = np.zeros_like(height)
    elif p.profile == "spherical_cap":
        _, dhdx, dhdy = spherical_cap_profile(display_x, display_y, p)
    else:
        dhdx = (gaussian_height(display_x + hs, display_y, p)
                - gaussian_height(display_x - hs, display_y, p)) / (2.0 * hs)
        dhdy = (gaussian_height(display_x, display_y + hs, p)
                - gaussian_height(display_x, display_y - hs, p)) / (2.0 * hs)
    hx, hy = dhdy, -dhdx
    w = 1.0 / np.sqrt(1.0 + hx * hx + hy * hy)
    normal = np.column_stack((-hx * w, -hy * w, w))
    zb = np.where(is_a, 0.5 * p.buckle, -0.5 * p.buckle)
    deformed = np.column_stack((base_x + zb * normal[:, 0],
                                base_y + zb * normal[:, 1],
                                height + zb * normal[:, 2]))

    def deform(positions):
        if positions.shape != deformed.shape:
            raise RuntimeError("Unexpected TBPLaS orbital ordering")
        positions[:] = deformed

    supercell = tb.SuperCell(cell, dim=(nx, ny, nz), pbc=(True, True, False),
                             orb_pos_modifier=deform)
    sample = tb.Sample(supercell)
    sample.init_orb_pos()
    sample.init_orb_eng()
    sample.init_hop()
    ii, jj = sample.hop_i, sample.hop_j
    nearest = (is_a[ii] != is_a[jj]) & (spin_up[ii] == spin_up[jj]) & (np.abs(sample.hop_v.real) > 0.5)
    distance = np.linalg.norm(sample.dr[nearest], axis=1)
    n2 = (sample.dr[nearest, 2] / distance) ** 2  # saved pzAxis='global'
    cutoff = 1.0 / (1.0 + np.exp((distance - 3.0 * p.d0) / 0.01))
    v_pi = -p.gamma0 * np.exp((2.0 / p.d0) * (p.d0 - distance)) * cutoff
    sample.hop_v[nearest] = (1.0 - n2) * v_pi

    bond_strain = distance / p.d0 - 1.0
    count = np.zeros(sample.num_orb)
    total = np.zeros(sample.num_orb)
    for ids in (ii[nearest], jj[nearest]):
        np.add.at(count, ids, 1.0)
        np.add.at(total, ids, bond_strain)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    strain = mean * p.d0 ** 2 / p.bond ** 2
    return Model(sample, p, (nx, ny, nz), center, height, strain,
                 np.column_stack((display_x[::2], display_y[::2])),
                 is_a[::2], nearest)


def cell_display(model: Model, site_values):
    """Average four physical sites per cell and rotate to the MATLAB display frame."""
    nx, ny = model.dim[:2]
    lattice = np.asarray(site_values).reshape(nx, ny, 8)[:, :, [0, 2, 4, 6]].mean(axis=2).T
    return np.flipud(lattice).T


def pseudofield(model: Model):
    """MATLAB hb_local_analysis cell gauge and centred periodic curl, in tesla."""
    sample, p = model.sample, model.params
    mask = model.nearest_mask
    ii, jj = sample.hop_i[mask], sample.hop_j[mask]
    dr = sample.dr[mask, :2]
    values = sample.hop_v[mask].real
    is_a = np.repeat(model.is_a, 2)
    sign_a = np.where(is_a[ii], 1.0, -1.0)
    directed = sign_a[:, None] * dr
    b = p.bond
    reference = b * np.array([[np.sqrt(3.0) / 2.0, -0.5],
                              [-np.sqrt(3.0) / 2.0, -0.5], [0.0, 1.0]])
    cls = np.argmax(directed @ reference.T, axis=1)
    total = np.zeros((sample.num_orb, 3))
    count = np.zeros_like(total)
    for k in range(3):
        take = cls == k
        for ids in (ii[take], jj[take]):
            np.add.at(total[:, k], ids, values[take])
            np.add.at(count[:, k], ids, 1.0)
    hopping = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    nx, ny = model.dim[:2]
    tcell = hopping.reshape(nx, ny, 8, 3)[:, :, [0, 2, 4, 6], :].mean(axis=2)
    t1, t2, t3 = (tcell[:, :, k].T for k in range(3))
    ax = (2.0 * t3 - t1 - t2) / 2.0
    ay = -np.sqrt(3.0) * (t1 - t2) / 2.0
    ax_paper = np.sqrt(3.0) * (t1 - t2) / 2.0
    ay_paper = ax
    vf = 1.5 * abs(p.hopping) * b * 1e-9 / HBAR_EV_S

    def curl(gx, gy):
        d_gy_dx = (np.roll(gy, -1, axis=1) - np.roll(gy, 1, axis=1)) / (2.0 * p.a)
        d_gx_dy = (np.roll(gx, -1, axis=0) - np.roll(gx, 1, axis=0)) / (6.0 * b)
        return (d_gy_dx - d_gx_dy) * 1e9 / vf

    display = lambda array: np.flipud(array).T
    x0, y0 = model.center
    xc = np.arange(nx) * p.a + 0.25 * p.a - x0
    yc = np.arange(ny) * 3.0 * b + 2.0 * b - y0
    return {"x": -yc[::-1], "y": xc,
            "B": display(curl(ax, ay)), "Bpaper": display(curl(ax_paper, ay_paper)),
            "T1": display(t1), "T2": display(t2), "T3": display(t3),
            "Ax": display(-ay), "Ay": display(ax), "vF": vf}


def line_sites(model: Model, *, length=9.0, points=25, angle_deg=0.0, sublattice="both"):
    """MATLAB hb_line_sites: nearest physical A/B atoms to points on a line."""
    theta = np.deg2rad(angle_deg)
    position = np.linspace(-length / 2.0, length / 2.0, points)
    target = np.column_stack((position * np.cos(theta), position * np.sin(theta)))
    if sublattice == "A":
        classes = [True]
    elif sublattice == "B":
        classes = [False]
    elif sublattice == "both":
        classes = [True, False]
    else:
        raise ValueError("sublattice must be 'A', 'B', or 'both'")
    selected = np.empty((points, len(classes)), dtype=int)
    for j, a_site in enumerate(classes):
        candidates = np.flatnonzero(model.is_a == a_site)
        xy = model.display_xy[candidates]
        for k, point in enumerate(target):
            selected[k, j] = candidates[np.argmin(np.sum((xy - point) ** 2, axis=1))]
    _, first = np.unique(selected, axis=0, return_index=True)
    keep = np.sort(first)
    return position[keep], selected[keep], target[keep]


def bubble_sites(model: Model, *, sublattice="both"):
    """Physical atoms within the bubble radius, in the undeformed display frame."""
    mask = np.linalg.norm(model.display_xy, axis=1) <= model.params.radius
    if sublattice == "A":
        mask &= model.is_a
    elif sublattice == "B":
        mask &= ~model.is_a
    elif sublattice != "both":
        raise ValueError("sublattice must be 'A', 'B', or 'both'")
    return np.flatnonzero(mask)


def bubble_a_sites(model: Model):
    """A-only selection retained for the original MATLAB reproduction notebook."""
    return bubble_sites(model, sublattice="A")


def lanczos_coeffs(ham, vector, depth):
    """MATLAB hb_lanczos_coeffs, with a sparse TBPLaS Hamiltonian."""
    v = np.asarray(vector, dtype=np.complex128)
    v = v / np.linalg.norm(v)
    vp = np.zeros_like(v)
    bp = 0.0
    alpha = np.empty(depth)
    beta = np.empty(depth)
    for m in range(depth):
        w = ham @ v
        alpha[m] = np.vdot(v, w).real
        w = w - alpha[m] * v - bp * vp
        beta[m] = np.linalg.norm(w)
        if beta[m] < 1e-10:
            return alpha[:m + 1], beta[:m + 1]
        vp, v, bp = v, w / beta[m], beta[m]
    return alpha, beta


def continued_fraction(alpha, beta, energy, eta):
    """Square-root terminator and Lorentz broadening from hb_cf_eval.m."""
    z = np.asarray(energy) + 1j * eta
    if beta[-1] < 1e-10:
        g = np.zeros_like(z)
    else:
        k = max(1, len(alpha) // 4)
        ainf = np.mean(alpha[-k:])
        binf = np.mean(beta[-k:])
        sq = np.sqrt((z - ainf) ** 2 - 4.0 * binf ** 2)
        g = (z - ainf - sq) / (2.0 * binf ** 2)
        flip = np.imag(g) > 0
        g[flip] = (z[flip] - ainf + sq[flip]) / (2.0 * binf ** 2)
    for a, b in zip(alpha[::-1], beta[::-1]):
        g = 1.0 / (z - a - b * b * g)
    return -np.imag(g) / np.pi


def spectral_density(ham, vector, energy, eta, depth):
    a, b = lanczos_coeffs(ham, vector, depth)
    return continued_fraction(a, b, energy, eta)


def site_ldos(ham, physical_site, energy, eta, depth):
    """MATLAB total LDOS = mean of up/down projections at one physical atom."""
    density = np.zeros_like(energy, dtype=float)
    for spin in (0, 1):
        v = np.zeros(ham.shape[0], dtype=np.complex128)
        v[2 * physical_site + spin] = 1.0
        density += 0.5 * spectral_density(ham, v, energy, eta, depth)
    return density


def line_ldos(ham, sites, energy, eta, depth):
    out = np.empty((len(energy), len(sites)))
    for k, row in enumerate(sites):
        out[:, k] = np.mean([site_ldos(ham, int(site), energy, eta, depth)
                             for site in row], axis=0)
    return out


def random_phase_vector(n, physical_sites, phases, spin):
    v = np.zeros(n, dtype=np.complex128)
    v[2 * physical_sites + spin] = phases / np.sqrt(len(physical_sites))
    return v


def roi_ldos(ham, physical_sites, energy, eta, depth, phases):
    """Stochastic bubble LDOS using the same phases for bubble/flat references."""
    draws = np.empty((len(energy), phases.shape[1]))
    for q in range(phases.shape[1]):
        curves = [spectral_density(
            ham, random_phase_vector(ham.shape[0], physical_sites, phases[:, q], spin),
            energy, eta, depth) for spin in (0, 1)]
        draws[:, q] = 0.5 * (curves[0] + curves[1])
    mean = draws.mean(axis=1)
    error = draws.std(axis=1, ddof=1) / np.sqrt(draws.shape[1]) if draws.shape[1] > 1 else np.full_like(mean, np.nan)
    return mean, error, draws


def total_dos(ham, energy, eta, depth, *, draws=8, seed=20260923):
    """DOS per spin orbital from a normalized full-cell random-phase trace."""
    rng = np.random.default_rng(seed)
    curves = np.empty((len(energy), draws))
    for q in range(draws):
        phase = np.exp(2j * np.pi * rng.random(ham.shape[0])) / np.sqrt(ham.shape[0])
        curves[:, q] = spectral_density(ham, phase, energy, eta, depth)
    return curves.mean(axis=1), curves.std(axis=1, ddof=1) / np.sqrt(draws), curves
