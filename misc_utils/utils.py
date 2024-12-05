import os

def create_virtual_folder(src_folder, virtual_folder):
    '''
    Creates a "virtual folder" by creating symbolic links for each file in the source folder (src_folder) to the 
    destination folder (virtual_folder). This function will walk through the directory tree of the source folder 
    and generate symlinks for all files, skipping files like .DS_Store. It will log the creation of each symlink, 
    as well as skipping already existing symlinks.

    Arguments:
        src_folder (str): The path to the source folder whose files need to be linked. This can be any directory on the 
        filesystem.
        virtual_folder (str): The path to the destination folder where the symlinks will be created. This folder will be 
        created if it doesn’t exist.

    Returns:
        List of strings: A list of log messages indicating the actions performed. Each log message will either note the 
        creation of a new symlink or the skipping of an existing symlink.
    '''
    os.makedirs(virtual_folder, exist_ok=True)
    logs = []
    for root, folders, files in os.walk(src_folder):
        for file in files:
            # Skip .DS_Store files
            if file == ".DS_Store":
                continue
            src_file = os.path.join(root, file)
            symlink_path = os.path.join(virtual_folder, file)
            if not os.path.exists(symlink_path):
                link = os.symlink(src_file, symlink_path)
                log = f"Created symlink: {symlink_path}"
            else:
                log = f"Skipping existing symlink: {symlink_path}"
            logs.append(log)
    return logs