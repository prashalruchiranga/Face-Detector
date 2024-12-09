import os
import numpy as np

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


def format_bbox(bbox):
    x1, y1, w, h = list(map(int, bbox.split()))
    formatted_bbox = list(map(str, [x1, x1+w, y1, y1+h]))
    return " ".join(formatted_bbox)


def format_labels(src, dest):
    ### Read src file
    with open(src, 'r') as f:
        bbox_anno = [line.rstrip("\n, ") for line in f.readlines()]    
    ### Create a dictionary such that it contains image_name:respective_annotations 
    img_indices = []
    n_bbxes = []
    for i in range(len(bbox_anno)):
        if bbox_anno[i].endswith((".jpg", ".jpeg", ".png")):
            img_indices.append(i)
            n_bbxes.append(int(bbox_anno[i+1]))
    annotations = {}
    collection = []
    for idx,n in list(zip(img_indices, n_bbxes)):
        img_name = bbox_anno[idx]
        res_annot = bbox_anno[idx+2 : idx+2+n]
        ### Remove blur, expression, illumination, invalid, occlusion and pose details. Keep x1, y1, w and h.
        res_annot = [" ".join(annot.split()[:4]) for annot in res_annot]
        ### Remove folder name from image name
        assert len(img_name.split('/')) == 2
        _, img_name = img_name.split('/')
        annotations[img_name] = res_annot
        ### Format bboxes
        if not(res_annot == []):
            class_id = 1
            for bbox in res_annot:
                formatted_bbox = format_bbox(bbox)
                ### Only class available in wider face dataset is face. Therefore class_id must equal to 1 in each row in ground truth csv file.
                collection.append(f"{img_name} {formatted_bbox} {str(class_id)}")
        else:
            class_id = 0
            collection.append(f"{img_name} {'0 0 0 0'} {str(class_id)}")
    ### Save to csv
    rows = [entry.split() for entry in collection]
    np.savetxt(fname=dest, X=rows, delimiter=',', fmt="%s", header='image_name xmin xmax ymin ymax class_id')
    log = f"Formatted file succesfully saved to {dest}"
    return log
    
    