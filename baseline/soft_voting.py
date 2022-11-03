import argparse
import torch
import pandas as pd
import os


def soft_voting(single_dir, output_dir, output_name, num_classes):
    
    file_names = os.listdir(single_dir)
    output_path = os.path.join(output_dir, output_name)
    ensemble_num = len(file_names)

    df_list = []
    for file_name in file_names:
        df_list.append(pd.read_csv(os.path.join(single_dir, file_name)))

    data_columns = []
    for i in range(num_classes):
        data_columns.append(str(i))

    image_id = df_list[0]['ImageID']

    data_tensors = []
    for i in range(ensemble_num):
        data_tensor = torch.tensor(df_list[i][data_columns].values)
        data_tensors.append(data_tensor[None, :])

    data_tensors = torch.cat(data_tensors, 0)    
    data_tensors = torch.sum(data_tensors, 0)

    ensemble_result = torch.argmax(data_tensors,1)
    ensemble_result = pd.Series(ensemble_result)

    ensemble_output = pd.concat([image_id, ensemble_result], axis=1)
    ensemble_output.columns = ['ImageID', 'ans']

    ensemble_output.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--single_dir', type=str, default='./soft_voting_candidates', help='directory path containing single-model result files')
    parser.add_argument('--output_dir', type=str, default='./output', help='directory path to save ensemble_output file')
    parser.add_argument('--output_name', type=str, default='ensemble_output.csv', help='ensemble_output file name')
    parser.add_argument('--num_classes', type=int, default=18, help='number of classes')

    args = parser.parse_args()

    single_dir = args.single_dir
    output_dir = args.output_dir
    output_name = args.output_name
    num_classes =args.num_classes

    os.makedirs(output_dir, exist_ok=True)

    soft_voting(single_dir, output_dir, output_name, num_classes)
    