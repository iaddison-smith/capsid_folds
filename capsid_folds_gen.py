import MDAnalysis as mda
from MDAnalysis import transformations
import numpy as np


# 5-folds (90 y-axis)
zika = mda.Universe('pqr/ZIKV_6CO8_aa_charge_vdw_addspace.pqr')
ts = zika.trajectory.ts
angle = 90 # Angle in degrees
ag = zika.atoms
d = [0,1,0] # direction of a custom axis of rotation from the provided point
rotated = transformations.rotate.rotateby(angle, direction=d, ag=ag)(ts)
rotation_axis = np.argmax(d)
direction_to_string = lambda x : 'x-axis' if x == 0 else 'y-axis' if x == 1 else 'z-axis'
zika.atoms.write('zika_rotation_angle%2d_%s.pqr'%(angle, direction_to_string(rotation_axis)))

# 2-folds (146.73 y-axis)
zika = mda.Universe('pqr/ZIKV_6CO8_aa_charge_vdw_addspace.pqr')
ts = zika.trajectory.ts
angle = 146.73 # Angle in degrees
ag = zika.atoms
d = [0,1,0] # direction of a custom axis of rotation from the provided point
rotated = transformations.rotate.rotateby(angle, direction=d, ag=ag)(ts)
rotation_axis = np.argmax(d)
direction_to_string = lambda x : 'x-axis' if x == 0 else 'y-axis' if x == 1 else 'z-axis'
zika.atoms.write('zika_rotation_angle%2d_%s.pqr'%(angle, direction_to_string(rotation_axis)))

# 3-folds (128.4 y-axis)
zika = mda.Universe('pqr/ZIKV_6CO8_aa_charge_vdw_addspace.pqr')
ts = zika.trajectory.ts
angle = 128.4 # Angle in degrees
ag = zika.atoms
d = [0,1,0] # direction of a custom axis of rotation from the provided point
rotated = transformations.rotate.rotateby(angle, direction=d, ag=ag)(ts)
rotation_axis = np.argmax(d)
direction_to_string = lambda x : 'x-axis' if x == 0 else 'y-axis' if x == 1 else 'z-axis'
zika.atoms.write('zika_rotation_angle%2d_%s.pqr'%(angle, direction_to_string(rotation_axis)))