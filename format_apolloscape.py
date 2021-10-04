import os
import numpy as np
import torch

dis_threshold = 3 # 3 meters
total_object_num = 1453 # in file 9051_7 the index is from 1 to 1453
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
            one_packaged_sequence[0] = {'trajectory': object_traj_sequence[0:observe_len], 
                                        'node_matrix': node_matrix_sequence[0:observe_len], 
                                        'graph': graph_sequence[0:observe_len]}

            one_packaged_sequence[1] = {'trajectory': object_traj_sequence[observe_len:total_len], 
                                        'node_matrix': node_matrix_sequence[observe_len:total_len], 
                                        'graph': graph_sequence[observe_len:total_len]}
            
            packaged_input_sequences.append(one_packaged_sequence[0])
            packaged_result_sequences.append(one_packaged_sequence[1])

        file_name = file_name.split('/')
        print("train set "+file_name[-1]+" generation finished!")        
        return packaged_input_sequences, packaged_result_sequences

    elif data_type == 'test':
        
        formatted_data = np.delete(data,[2,5,6,7,8,9],axis=1)
        
        return torch.zeros(1,) 

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
        for train_file_name in train_file_names:
            file_name = DATA_DIR_TRAIN + train_file_name 
            generated_one_input, generated_one_result = generate_data(file_name, data_type)
            generated_input += generated_one_input
            generated_result += generated_one_result

        to_save_input_txt = DATA_DIR_TRAIN + 'formatted/train_input.npy'
        to_save_result_txt = DATA_DIR_TRAIN + 'formatted/train_result.npy'        

        print(len(generated_input))
        # for input in generated_input:
        #     print(input['trajectory'].size())
        #     print(input['node_matrix'].size())

        save_to_file( generated_input, to_save_input_txt)
        save_to_file( generated_result, to_save_result_txt)

    #     save_to_text(formatted_data, to_save_txt)


    if data_type == 'val':

        file = DATA_DIR + train_file_names[8]

        generated_data = generate_data(file, data_type)


        to_save_txt = './data/APOL/val/valSet0.npy'

        save_to_file( generated_data, to_save_txt)


    if data_type == 'test':

        file = DATA_DIR_TEST + test_file_names[0]

        generated_data = generate_data(file, data_type)

        print(generated_data[0:100])

        to_save_txt = './resources/data/' + 'APOL/test/testSet0.npy'

        save_to_file( generated_data, to_save_txt)

'''
Instructions for directory structure:
1. Download the dataset from the link provided in the README.md
2. Unzip the downloaded files sample_trajectory.zip and prediction_test.zip
3. Follow below format

dir = folder_where_unzipped_apolloscape_data_is_present
DATA_DIR = folder_where_unzipped_apolloscape_data_is_present + '/sample_trajectory/asdt_sample_ trajectory/'
DATA_DIR_TEST = folder_where_unzipped_apolloscape_data_is_present + '/prediction_test/'
'''
dir = '/home/mount/GCN-lstm/data/Aplloscape'
DATA_DIR_TRAIN = dir + '/prediction_train/'
DATA_DIR_TEST = dir + '/prediction_test/'

data_type = 'train' #train, test

apolloscape_to_formatted(DATA_DIR_TRAIN, DATA_DIR_TEST, data_type)