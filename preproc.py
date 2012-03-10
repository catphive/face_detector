import faces

from itertools import izip

def main():
    data = []
    faces.load_data_dir('Face16', True, data, 20)
    faces.load_data_dir('Nonface16', False, data, 20)

    with open("train.json", "w") as train_file:
        faces.dump(data, train_file)
    
if __name__ == "__main__":
    main()
