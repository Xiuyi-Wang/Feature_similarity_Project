import os

def check_files_exist(files):
    """
    if all the files exist, return True, else return False
    :param files: a list, including all the files that need to be tested
    :return:
    """
    num_file_not_exist = 0
    for file in files:
        if not os.path.exists(file):
           num_file_not_exist +=1
           break
    if num_file_not_exist!=0:
        return False
    else:
        return True