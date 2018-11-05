import tables
import numpy as np
from arguments import get_args
args = get_args()
import cv2
import scipy.ndimage as ndimage

terminal_states_f = tables.open_file(
    '{}/terminal_states.h5'.format(
        args.save_dir,
    ),
    mode='r',
)

num_data_perframe = 1000

def fixation_to_salmap_2d(fixation):
    salmap = np.zeros(
        (args.episode_length_limit*2+1,args.episode_length_limit*2+1)
    )
    for fixation_count in range(fixation.shape[0]):
        salmap[
            np.clip(
                int(fixation[fixation_count,0]),
                -args.episode_length_limit,
                +args.episode_length_limit
            )+args.episode_length_limit,
            np.clip(
                int(fixation[fixation_count,1]),
                -args.episode_length_limit,
                +args.episode_length_limit
            )+args.episode_length_limit
        ] += 1.0
    salmap = ndimage.gaussian_filter(salmap, sigma=(10, 10), order=0)
    # salmap = salmap[:,int(config['cordi']['lon']['range']):int(config['cordi']['lon']['range'])*2]
    salmap = salmap / np.max(salmap)
    return salmap

log_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
log_fps = 10
'''log everything with video'''
videoWriter = cv2.VideoWriter(
    '{}/terminal_states.avi'.format(
        args.save_dir,
    ),
    log_fourcc,
    log_fps,
    (args.episode_length_limit*2+1,args.episode_length_limit*2+1),
)

frame_i = 0
while True:
    print('[{}]'.format(
        frame_i,
    ))
    if (frame_i+num_data_perframe >= terminal_states_f.root.data.shape[0]):
        terminal_states_f.close()
        videoWriter.release()
        break
    salmap = (fixation_to_salmap_2d(
        fixation = terminal_states_f.root.data[frame_i:frame_i+num_data_perframe,:],
    )*255.0).astype(np.uint8)
    salmap = cv2.cvtColor(salmap, cv2.cv2.COLOR_GRAY2RGB)
    videoWriter.write(salmap.astype(np.uint8))
    frame_i += 1
