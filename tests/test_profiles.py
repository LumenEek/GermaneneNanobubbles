"""Geometry and spectral checks independent of the plotting notebooks."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import (ExportParams, build_model, spherical_cap_profile,
                   bubble_height, spectral_density, pseudofield)


class Profiles(unittest.TestCase):
    def test_sphere_and_derivatives(self):
        for height in (0.5, 1., 1.5, 2.):
            p = ExportParams(profile="spherical_cap", height=height)
            x = np.array([0., 0.7, 1.8, 2.7])
            y = np.array([0., 0.4, -0.6, 0.2])
            z, hx, hy = spherical_cap_profile(x, y, p)
            sphere = (p.radius**2 + height**2)/(2*height)
            np.testing.assert_allclose(x*x+y*y+(z+sphere-height)**2, sphere**2)
            eps = 1e-5
            dx = (bubble_height(x+eps,y,p)-bubble_height(x-eps,y,p))/(2*eps)
            dy = (bubble_height(x,y+eps,p)-bubble_height(x,y-eps,p))/(2*eps)
            np.testing.assert_allclose(hx,dx,atol=1e-9)
            np.testing.assert_allclose(hy,dy,atol=1e-9)
            z,hx,hy = spherical_cap_profile([0,3,4],0,p)
            np.testing.assert_allclose(z,[height,0,0],atol=1e-14)
            np.testing.assert_array_equal(hx,[0,0,0])
            np.testing.assert_array_equal(hy,[0,0,0])

    def test_invalid_caps(self):
        for h,r in ((-1,3),(3,3),(4,3),(1,0),(np.nan,3)):
            with self.assertRaises(ValueError):
                spherical_cap_profile(0,0,ExportParams(height=h,radius=r))

    def test_flat_and_periodic_hamiltonian(self):
        p = ExportParams(profile="spherical_cap", height=0, lfac=3, lambda_r2_mev=0)
        cap = build_model(p)
        flat = build_model(replace(p,profile="gaussian"),flat=True)
        np.testing.assert_array_equal(cap.sample.orb_pos,flat.sample.orb_pos)
        np.testing.assert_array_equal(cap.sample.hop_v,flat.sample.hop_v)
        np.testing.assert_allclose(pseudofield(cap)['Bpaper'],0,atol=1e-9)
        cap = build_model(replace(p,height=2))
        ham = cap.hamiltonian(); ham.eliminate_zeros()
        self.assertEqual((ham-ham.getH()).nnz,0)
        self.assertEqual(ham[::2,1::2].nnz,0)
        ii,jj = cap.sample.hop_i[cap.nearest_mask],cap.sample.hop_j[cap.nearest_mask]
        count = np.bincount(np.r_[ii,jj],minlength=ham.shape[0])
        np.testing.assert_array_equal(count,3)  # both in-plane boundaries periodic
        self.assertTrue(np.isfinite(pseudofield(cap)['Bpaper']).all())

    def test_recursion_against_exact_resolvent(self):
        h = np.array([[0.2,0.3+0.1j],[0.3-0.1j,-0.4]])
        energy = np.linspace(-1,1,101); eta=0.01
        rho = spectral_density(csr_matrix(h),[1,0],energy,eta,10)
        exact = np.array([-np.linalg.inv((e+1j*eta)*np.eye(2)-h)[0,0].imag/np.pi
                          for e in energy])
        np.testing.assert_allclose(rho,exact,atol=1e-11)


if __name__ == "__main__":
    unittest.main()
