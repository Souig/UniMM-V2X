import os
import numpy as np
import pickle
from os.path import join
OUT_DIR = 'data/infos/v2x-sim/'
def gengrate_map(map_root):
    map_infos = {}
    for file_name in os.listdir(map_root):
        if '.npz' in file_name:
            map_info = dict(np.load(join(map_root,file_name), allow_pickle=True)['arr'])
            town_name = file_name.split('_')[0]
            print(town_name)
            map_infos[town_name] = {} 
            lane_points = []
            lane_types = []
            lane_sample_points = []
            trigger_volumes_points = []
            trigger_volumes_types = []
            trigger_volumes_sample_points = []
            for road_id, road in map_info.items():
                for lane_id, lane in road.items():
                    if lane_id == 'Trigger_Volumes':
                        for single_trigger_volume in lane:
                            points = np.array(single_trigger_volume['Points'])
                            points[:,1] *= -1 #left2right
                            trigger_volumes_points.append(points)
                            trigger_volumes_sample_points.append(points.mean(axis=0))
                            trigger_volumes_types.append(single_trigger_volume['Type'])
                    else:
                        for single_lane in lane:
                            points = np.array([raw_point[0] for raw_point in single_lane['Points']])
                            points[:,1] *= -1
                            lane_points.append(points)
                            lane_types.append(single_lane['Type'])
                            lane_lenth = points.shape[0]
                            if lane_lenth % 50 != 0:
                                devide_points = [50*i for i in range(lane_lenth//50+1)]
                            else:
                                devide_points = [50*i for i in range(lane_lenth//50)]
                            devide_points.append(lane_lenth-1)
                            lane_sample_points_tmp = points[devide_points]
                            lane_sample_points.append(lane_sample_points_tmp)
            map_infos[town_name]['lane_points'] = lane_points
            map_infos[town_name]['lane_sample_points'] = lane_sample_points
            map_infos[town_name]['lane_types'] = lane_types
            map_infos[town_name]['trigger_volumes_points'] = trigger_volumes_points
            map_infos[town_name]['trigger_volumes_sample_points'] = trigger_volumes_sample_points
            map_infos[town_name]['trigger_volumes_types'] = trigger_volumes_types
    with open(join(OUT_DIR,'v2xsim_map_infos_temporal.pkl'),'wb') as f:
        pickle.dump(map_infos,f)

if __name__ == '__main__':
    map_root = 'dataset/maps'
    gengrate_map(map_root)
    print('generate map infos done!')
    # with open(join(OUT_DIR,'v2xsim_map_infos_temporal.pkl'),'rb') as f:
    #     map_infos = pickle.load(f)
    # print(map_infos['Town01']['lane_points'][0].shape)
        
