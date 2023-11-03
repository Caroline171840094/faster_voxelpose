from lib.utils.hi4d_cam_utils import read_cameras
import json
root = 'data/Hi4D/Hi4D_new/pair00_taichi00'
cams_dict = read_cameras(root)

cams_new = {}
cam_idx = 0
for cam in cams_dict.keys():
    
    cam_info = {}
    cam_info['R'] = cams_dict[cam]['R'].tolist()
    cam_info['T'] = (cams_dict[cam]['T']*1000).tolist() 
    cam_info['fx'] = cams_dict[cam]['K'][0, 0]
    cam_info['fy'] = cams_dict[cam]['K'][1, 1]
    cam_info['cx'] = cams_dict[cam]['K'][0, 2]
    cam_info['cy'] = cams_dict[cam]['K'][1, 2]
    cam_info['k'] = cams_dict[cam]['dist'][0, [0, 1, 4]].reshape(3, 1).tolist()
    cam_info['p'] = cams_dict[cam]['dist'][0, [2, 3]].reshape(2, 1).tolist()
    cams_new[str(cam_idx)] = cam_info
    cam_idx += 1
    
with open('data/Hi4D/Hi4D_new/calibration_hi4d.json', 'w') as f:
    # write the dictionary to the file in JSON format
    json.dump(cams_new, f)