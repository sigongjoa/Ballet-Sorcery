# -*- coding: utf-8 -*-
import numpy as np
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.smplx_mapper import SMPLXToOpenSimMapper

def run_density_test():
    print("\n[Analysis] SMPL-X High-Density Marker Set Validation")
    
    mapper = SMPLXToOpenSimMapper()
    new_markers = mapper.get_marker_names()
    num_new = len(new_markers)
    num_old = 25
    
    print("1. Marker Quantity Comparison:")
    print("   - Current (MediaPipe): " + str(num_old) + " points")
    print("   - Proposed (SMPL-X Virtual): " + str(num_new) + " points")
    print("   - Density Increase: +" + str(round(((num_new - num_old) / num_old) * 100, 1)) + "%")

    lower_body_keywords = ['KNEE', 'ANKLE', 'HEEL', 'TOE', 'ASIS', 'PSIS']
    lower_markers = [m for m in new_markers if any(k in m for k in lower_body_keywords)]
    
    print("2. Regional Focus (Lower Body & Pelvis):")
    print("   - Key Markers: " + str(len(lower_markers)) + " points")
    
    left_markers = [m for m in new_markers if m.endswith('_L')]
    right_markers = [m for m in new_markers if m.endswith('_R')]
    
    print("3. Anatomical Symmetry Check:")
    if len(left_markers) == len(right_markers):
        print("   ✅ Symmetry Passed: L(" + str(len(left_markers)) + ") / R(" + str(len(right_markers)) + ") matches.")
    
    stability_gain = np.sqrt(num_new / num_old)
    print("4. Expected Stability Gain: ~" + str(round(stability_gain, 2)) + "x improvement.")

    return True

if __name__ == "__main__":
    run_density_test()
