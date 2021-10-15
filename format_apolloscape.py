import os
import numpy as np
import torch

dis_threshold = 3 # 3 meters

total_object_num = 1453 # in file 9051_7 the index is from 1 to 1453
test_object_num = 1000 # in file 9051_7 the index is from 1 to 1453

feature_size = 10 # node feature size: frame_id,object_id,object_type,x,y,z,l,w,h,orientation

observe_len = 6 # 3 sec
predict_len = 6 # 3 sec
total_len = observe_len + predict_len


def generate_data_from_txt(files_path):
    '''
    to convert the data from all txt files into one large numpy array
    :param files_path: filepath to load
    :return: numpy array of the data from txt file
    '''
    data = np.loadtxt(files_path)
    return data

def save_to_file(data,files_path_to_sv):
    '''
    to save the obtained merged numpy array to .npy
    :param data: large numpy array
    :param files_path_to_sv: file path where this .npy file will be saved
    :return: None
    '''
    np.save(files_path_to_sv, data, allow_pickle=True)

def generate_data(file_name, data_type):

    '''
    to create the formatted data from the apolloscape data
    :param file: file in which the unformatted data is present
    :param data: whether it is train or val or test data
    :return: formatted data
    '''
    
    print(file_name)
    
    data = generate_data_from_txt(file_name)
    data = torch.from_numpy(data)

    if data_type == 'train':
        
        frame_ids = []
        for line in data:
            if frame_ids.count(int(line[0])) == 0:
                frame_ids.append(int(line[0]))

        per_frame_data = []
        counter = 0
        for frame_id in frame_ids:
            one_frame_data = []

            for line in data[counter:]:
                if line[0] == frame_id:
                    one_frame_data.append(line)
                    counter = counter + 1
                else:
                    break
            
            per_frame_data.append(torch.stack(one_frame_data,dim=0))
        
        per_frame_graphs = []
        for one_frame_data in per_frame_data:
            one_frame_graph = []

            for object_i in one_frame_data:
                for object_j in one_frame_data:
                    dis = (object_i[3]-object_j[3])**2 + (object_i[4]-object_j[4])**2
                    if dis <= dis_threshold**2:
                        one_frame_graph.append(torch.IntTensor([object_i[1],object_j[1]]))
                            
            per_frame_graphs.append(torch.stack(one_frame_graph,dim=1))
        
        cleaned_per_frame_objects = []
        for frame_id in frame_ids:
            if frame_id + total_len > len(frame_ids):
                break
            
            one_frame_objects = []
            for frame_id_cur in range(frame_id, frame_id+total_len):
                one_frame_data = per_frame_data[frame_id_cur]
                for one_frame_line in one_frame_data:
                    if one_frame_line[0] != frame_id_cur:
                        print("WARNING: generation wrong at current dataset")
                        exit()
                    one_frame_objects.append(int(one_frame_line[1]))
            
            cleaned_one_frame_objects = []
            for one_frame_object in one_frame_objects:
                if one_frame_objects.count(one_frame_object) == total_len:
                    if cleaned_one_frame_objects.count(one_frame_object) == 0:
                        cleaned_one_frame_objects.append(one_frame_object)

            cleaned_per_frame_objects.append(cleaned_one_frame_objects)
        
        cleaned_objects_sequence = []
        for frame_id in frame_ids:
            if frame_id + total_len > len(frame_ids):
                break

            cleaned_one_frame_objects = cleaned_per_frame_objects[frame_id]
            for object_idx in cleaned_one_frame_objects:
                object_traj_sequence = torch.zeros(total_len, feature_size)
                for i in range(total_len):
                    cur_frame_data = per_frame_data[frame_id+i]
                    for cur_frame_line in cur_frame_data:
                        if cur_frame_line[1] == object_idx:
                            object_traj_sequence[i] = cur_frame_line
                            break
                cleaned_objects_sequence.append(object_traj_sequence)
        
        per_frame_node_matrix = []
        for frame_id in frame_ids:
            one_frame_node_matrix = torch.zeros(total_object_num, feature_size)
            one_frame_data = per_frame_data[frame_id]
            for one_frame_line in one_frame_data:
                if one_frame_line[0] != frame_id:
                    print("WARNING: generation wrong at current dataset 2")
                    exit()
                one_frame_node_matrix[int(one_frame_line[1])-1] = one_frame_line
            
            per_frame_node_matrix.append(one_frame_node_matrix)

        packaged_input_sequences = []
        packaged_result_sequences = []
        for object_traj_sequence in cleaned_objects_sequence:
            object_idx = int(object_traj_sequence[0][1])
            frame_idx = int(object_traj_sequence[0][0])

            node_matrix_sequence = torch.stack(per_frame_node_matrix[frame_idx: frame_idx+total_len] ,dim=0)
            graph_sequence = per_frame_graphs[frame_idx: frame_idx+total_len]

            one_packaged_sequence = {}
            one_packaged_sequence[0] = {
                'trajectory': object_traj_sequence[0:observe_len], 
                'node_matrix': node_matrix_sequence[0:observe_len], 
                'graph': graph_sequence[0:observe_len]
            }

            one_packaged_sequence[1] = {
                'trajectory': object_traj_sequence[observe_len:total_len], 
                'node_matrix': node_matrix_sequence[observe_len:total_len], 
                'graph': graph_sequence[observe_len:total_len]
            }
            
            packaged_input_sequences.append(one_packaged_sequence[0])
            packaged_result_sequences.append(one_packaged_sequence[1])

        file_name = file_name.split('/')
        print("train set "+file_name[-1]+" generation finished!")        
        return packaged_input_sequences, packaged_result_sequences

    elif data_type == 'test':
        
        per_sequence_frame_ids = []
        one_sequence_frame_ids = []
        begin_frame_id = 0
        cur_frame_id = 0
        for line in data:
            idx_ofnow = int(line[0])
            if idx_ofnow - cur_frame_id != 1:
                if idx_ofnow == cur_frame_id:
                    continue
                else:
                    begin_frame_id = idx_ofnow
                    cur_frame_id = idx_ofnow
                    per_sequence_frame_ids.append(one_sequence_frame_ids.copy())
                    one_sequence_frame_ids.clear()

                    one_sequence_frame_ids.append(cur_frame_id)
            else:
                if idx_ofnow - begin_frame_id == 6:
                    begin_frame_id = idx_ofnow
                    cur_frame_id = idx_ofnow
                    per_sequence_frame_ids.append(one_sequence_frame_ids.copy())
                    one_sequence_frame_ids.clear()

                    one_sequence_frame_ids.append(cur_frame_id)
                else:
                    if one_sequence_frame_ids.count(idx_ofnow) == 0:
                        one_sequence_frame_ids.append(idx_ofnow)
                        cur_frame_id = idx_ofnow
        
        per_sequence_frame_ids.pop(0)
        per_sequence_frame_ids.append(one_sequence_frame_ids.copy())


        per_sequence_lines = []
        per_sequence_object_ids = []
        line_counter = 0
        for one_sequence_frame_ids in per_sequence_frame_ids:
            one_sequence_lines = []
            one_sequence_object_ids = []
            for frame_id in one_sequence_frame_ids:
                for line in data[line_counter:]:
                    frame_id_ofnow = int(line[0])
                    object_id_ofnow = int(line[1])
                    if frame_id_ofnow == frame_id:
                        one_sequence_lines.append(line)
                        if one_sequence_object_ids.count(object_id_ofnow) == 0:
                            one_sequence_object_ids.append(object_id_ofnow)
                        line_counter += 1
                    else:
                        break

            per_sequence_lines.append(torch.stack(one_sequence_lines,dim=0))
            per_sequence_object_ids.append(one_sequence_object_ids)

        per_frame_lines = []
        for one_sequence_frame_ids,one_sequence_lines in zip(per_sequence_frame_ids,per_sequence_lines):
            frame_ids = one_sequence_lines[:,0].tolist()
            chunk_list = [ frame_ids.count(i) for i in one_sequence_frame_ids ]
            many_frame_lines = torch.split(one_sequence_lines,chunk_list)
            for one_frame_lines in many_frame_lines:
                per_frame_lines.append(one_frame_lines)
        
        per_frame_graphs = []
        per_frame_node_matrix = []
        per_frame_min_idx = []
        frame_counter = 0
        for one_sequence_lines in per_sequence_lines:
            min_idx = torch.min(one_sequence_lines[:,1])

            for i in range(observe_len):
                one_frame_graph = []
                one_frame_node_matrix = torch.zeros(test_object_num,feature_size)

                one_frame_lines = per_frame_lines[frame_counter+i]
                for object_i in one_frame_lines:
                    one_frame_node_matrix[int(object_i[1]-min_idx)] = object_i
                    for object_j in one_frame_lines:
                        dis = (object_i[3]-object_j[3])**2 + (object_i[4]-object_j[4])**2
                        if dis <= dis_threshold**2:
                            one_frame_graph.append(torch.IntTensor([object_i[1]-min_idx,object_j[1]-min_idx]))
                            # one_frame_graph.append(torch.IntTensor([object_i[1],object_j[1]]))
                per_frame_graphs.append(torch.stack(one_frame_graph,dim=1))
                per_frame_node_matrix.append(one_frame_node_matrix)
                per_frame_min_idx.append(min_idx)

            frame_counter += observe_len
        
        objects_trajectory = []
        for one_sequence_object_ids,one_sequence_frame_ids,one_sequence_lines in zip(per_sequence_object_ids,per_sequence_frame_ids,per_sequence_lines):
            one_sequence_trajectory = torch.zeros(len(one_sequence_object_ids),observe_len,feature_size)
            for line in one_sequence_lines:
                object_id = one_sequence_object_ids.index(int(line[1]))
                frame_id = one_sequence_frame_ids.index(int(line[0]))
                one_sequence_trajectory[object_id,frame_id,:] = line

            for trajectory in one_sequence_trajectory:
                zero_flag = False
                for step in trajectory:
                    if int(step[0]) == 0:
                        zero_flag = True
                        break
                if zero_flag:
                    calibration = 0
                    for idx,step in enumerate(trajectory):
                        if int(step[0]) != 0:
                            calibration = step[0] - idx
                            break
                    for idx,step in enumerate(trajectory):
                        step[0] = calibration + idx

                objects_trajectory.append(trajectory)

        packaged_input_sequences = []
        for object_trajectory in objects_trajectory:
            one_packaged_sequence = {}
            begin_frame_id = int(object_trajectory[0][0])
            sequence_id = 0
            for i, one_sequence_frame_ids in enumerate(per_sequence_frame_ids):
                if begin_frame_id == one_sequence_frame_ids[0]:
                    sequence_id = i
                    break

            current_sequence_graphs = per_frame_graphs[sequence_id*6:(sequence_id+1)*6]
            current_sequence_node_matrixs = per_frame_node_matrix[sequence_id*6:(sequence_id+1)*6]
            current_sequence_min_idx = per_frame_min_idx[sequence_id*6:(sequence_id+1)*6]
            one_packaged_sequence = {
                'trajectory': object_trajectory, 
                'node_matrix': current_sequence_node_matrixs, 
                'graph': current_sequence_graphs,
                'min_idx' : current_sequence_min_idx
            }
            packaged_input_sequences.append(one_packaged_sequence)

        file_name = file_name.split('/')
        print("test set "+file_name[-1]+" generation finished!")        
        return packaged_input_sequences

def apolloscape_to_formatted(DATA_DIR_TRAIN, DATA_DIR_TEST, data_type):

    
    train_file_names = []
    test_file_names = []

    for file in sorted(os.listdir(DATA_DIR_TRAIN)):
        if file.endswith('.txt'):
            train_file_names.append(file)


    for file in sorted(os.listdir(DATA_DIR_TEST)):
        if file.endswith('.txt'):
            test_file_names.append(file)

    if data_type == 'train':

        generated_input = []
        generated_result = []
        # counter = 5
        for train_file_name in train_file_names:
            file_name = DATA_DIR_TRAIN + train_file_name 
            generated_one_input, generated_one_result = generate_data(file_name, data_type)
            generated_input += generated_one_input
            generated_result += generated_one_result
            # counter -= 1
            # if counter < 0:
            #     break
        to_save_input_txt = DATA_DIR_TRAIN + 'formatted/train_input_sub.npy'
        to_save_result_txt = DATA_DIR_TRAIN + 'formatted/train_result_sub.npy'        

        print(len(generated_input))
        # for input in generated_input:
        #     print(input['trajectory'].size())
        #     print(input['node_matrix'].size())

        save_to_file( generated_input, to_save_input_txt)
        save_to_file( generated_result, to_save_result_txt)

    #     save_to_text(formatted_data, to_save_txt)


    if data_type == 'test':

        generated_input = []

        for test_file_name in test_file_names:
            file_name = DATA_DIR_TEST + test_file_name 
            generated_one_input = generate_data(file_name, data_type)
            generated_input += generated_one_input
            
        to_save_input_txt = DATA_DIR_TEST + 'formatted/test_input.npy'
        
        print(len(generated_input)) #414
        # for input in generated_input:
        #     print(input['trajectory'].size())
        #     print(input['node_matrix'].size())

        save_to_file( generated_input, to_save_input_txt)

'''
Instructions for directory structure:
1. Download the dataset from the link provided in the README.md
2. Unzip the downloaded files sample_trajectory.zip and prediction_test.zip
3. Follow below format

dir = folder_where_unzipped_apolloscape_data_is_present
DATA_DIR = folder_where_unzipped_apolloscape_data_is_present + '/sample_trajectory/asdt_sample_ trajectory/'
DATA_DIR_TEST = folder_where_unzipped_apolloscape_data_is_present + '/prediction_test/'
'''
dir = '/home/mount/GCN-lstm/data/Apolloscape'
DATA_DIR_TRAIN = dir + '/prediction_train/'
DATA_DIR_TEST = dir + '/prediction_test/'

data_type = 'test' #train, test

apolloscape_to_formatted(DATA_DIR_TRAIN, DATA_DIR_TEST, data_type)