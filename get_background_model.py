import numpy as np
import cv2
import natsort
import os

def main():

    img_dir = '/datasets/Real_sense/mod_data_081218/complex1/10'
    num_frames_to_use = 100

    # Load depth_frames 
    depth_frames_list = [f for f in os.listdir(img_dir) if f.endswith(".dat")]
    depth_frames_list = natsort.natsorted(depth_frames_list)

    depth_frames = []
    for frame_idx in range(num_frames_to_use):
        image_name = os.path.join(img_dir, depth_frames_list[frame_idx])
        depth_map = extract_depth_frame(image_name, 640, 480)
        print('data min, max:', np.amin(depth_map), np.amax(depth_map))
        depth_frames.append(depth_map)

        image = depth_map_to_image(depth_map)

        cv2.imshow('depth', image)

        k = cv2.waitKey(0) & 0xFF
        if k ==27:
            break

    background = extract_background(depth_frames)
    print('background min, max:', np.amin(background), np.amax(background))

    # save background
    serializeMatbin(background, os.path.join(img_dir, 'background.matbin'))

    # Display background
    background_img = depth_map_to_image(background)
    cv2.imshow('background', background_img)

    # restore background
    background_restored = deserializeMatbin(os.path.join(img_dir, 'background.matbin'))
    print('background_restored min, max, argmin:', np.amin(background_restored),
          np.amax(background_restored), np.argmin(background_restored))
    background_restored_img = depth_map_to_image(background_restored)
    cv2.imshow('background_restored', background_restored_img)
    cv2.waitKey(0)



def extract_depth_frame(filename, cols, rows):
    f = open(filename, "rb")
    output = np.fromfile(filename, dtype='i2', count=cols * rows).reshape(rows, cols, 1)

    return output

def depth_map_to_image(depth_map):
    depth_map = depth_map.astype(np.float)
    img = cv2.normalize(depth_map, depth_map, 0, 1, cv2.NORM_MINMAX)
    img = np.array(img * 255, dtype = np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)    

    return img

def extract_background(depth_frames):
    depth_frames = np.squeeze(depth_frames)
    background = np.median(depth_frames, axis=0)
    background = background.astype(np.int16)

    return background

def deserializeMatbin(filename):
    
    f = open(filename, "rb")
    
    cols = 640
    rows = 480
    output = np.fromfile(filename, dtype='i2', count=cols * rows).reshape(rows, cols, 1)

    return output

def serializeMatbin(depth_file_16b, output_file):
    
    f = open(output_file, "wb")
    depth_file_16b.astype('int16').tofile(f)

    f.close()

if __name__ == '__main__':
    main()