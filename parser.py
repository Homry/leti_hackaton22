
def parse(path_to_data):
    data_same_person = []
    data_dif_person = []
    with open(path_to_data, 'r') as f:
        line = f.read()
        line = line.split('\n')
        line = [i.split('\t') for i in line]
        [data_same_person.append(i) if len(i) == 3 else data_dif_person.append(i) for i in line]
    data_same_person_res = [[f'lfw/{i[0]}/{i[0]}_000{i[1]}.jpg' if len(i[1]) == 1 else f'lfw/{i[0]}/{i[0]}_00{i[1]}',
                             f'lfw/{i[0]}/{i[0]}_000{i[2]}' if len(i[2]) == 1 else f'lfw/{i[0]}/{i[0]}_00{i[2]}.jpg',
                             1.0] for i in data_same_person]
    del (data_dif_person[-1])
    data_dif_person_res = [[f'lfw/{i[0]}/{i[0]}_000{i[1]}.jpg' if len(i[1]) == 1 else f'lfw/{i[0]}/{i[0]}_00{i[1]}',
                            f'lfw/{i[2]}/{i[2]}_000{i[3]}.jpg' if len(i[3]) == 1 else f'lfw/{i[2]}/{i[2]}_00{i[3]}',
                            0.0] for i in data_dif_person]
    return data_same_person_res, data_dif_person_res


if __name__ == "__main__":
    d1, d2 = parse('./pairsDevTrain.txt')



