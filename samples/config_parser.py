import yaml
from pathlib import Path

"""
example of a yaml file

- mode : image
  images_path: ../data/imgs_MOV03478
  groundtruth_path: ../layout/mov03478.csv
  model_name: efficientdet_d0
  adapter_name: AdapterDetectionTask
  path_classes : ../configs/efficientdet_d0/classes_coco90.txt
  path_weights : ../configs/efficientdet_d0/efficientdet_d0_frozen.pb
  path_config : ../configs/efficientdet_d0/efficientdet_d0.pbtxt
  write_path: data.csv
  confidence: 0.3
  nms_threshold: 0.4
  scale: 0.00392156
  size: 512 512
  mean: 123.675 116.28 103.53
  swapRB: 1
  silent_mode: 0
"""

def parse_yaml_file(yaml_file):
    
    with open(yaml_file, 'r', encoding = 'utf-8') as fh:
        parameters = yaml.safe_load(fh)
    parameters = parameters[0]
    
    mode = parameters.get('mode')

    if mode == None:
        raise ValueError('mode is not specified. This parameter is required.')
    
    if mode != 'image' and mode != 'video':
        raise ValueError('The mode is specified incorrectly.') 
    
    if mode == 'image' and parameters.get('images_path') == None:
        raise ValueError('In image mode, the images_path parameter is required.') 
    
    if mode == 'video' and parameters.get('video_path') == None:
        raise ValueError('In video mode, the video_path parameter is required.') 

    if parameters.get('model_name') == None:
        parameters.update({'model_name' : None}) 
       
    list_adapter = ['AdapterYOLO', 'AdapterYOLOTiny', 'AdapterDetectionTask', 'AdapterFasterRCNN']
    if not (parameters.get('adapter_name') in list_adapter):
        raise ValueError('The adapter is specified incorrectly.\n List of acceptable models: \'AdapterYOLO\', \'AdapterYOLOTiny\', \'AdapterDetectionTask\', \'AdapterFasterRCNN\'')

    if parameters.get('path_classes') == None:
        raise ValueError('path_classes is not specified. This parameter is required.')
    
    if parameters.get('path_weights') == None:
        parameters.update({'path_weights' : None}) 
    
    if parameters.get('path_config') == None:
        parameters.update({'path_config' : None}) 
    
    if parameters.get('confidence') == None:
        parameters.update({'confidence' : 0.3})
    else:
        parameters['confidence'] = float(parameters['confidence'])
        
    if parameters.get('nms_threshold') == None:
        parameters.update({'nms_threshold' : 0.4})
    else:
        parameters['nms_threshold'] = float(parameters['nms_threshold'])
        
    if parameters.get('scale') == None:
        parameters.update({'scale' : 1.0})
    else:
        parameters['scale'] = float(parameters['scale'])
        
    if parameters.get('size') == None:
        parameters.update({'size' : [0, 0]})
    else:
        parameters['size'] = list(map(int, parameters['size'].split(' ')))
    
    if parameters.get('mean') == None:
        parameters.update({'mean' : [0.0, 0.0, 0.0]})
    else:
        parameters['mean'] = list(map(float, parameters['mean'].split(' ')))

    if parameters.get('swapRB') == None:
        parameters.update({'swapRB' : False})
    else:
        parameters['swapRB'] = bool(parameters['swapRB'])
    
    if parameters.get('groundtruth_path') == None:
        parameters.update({'groundtruth_path' : None})
        
    if parameters.get('write_path') == None:
        parameters.update({'write_path' : None})
    else:
        parameters['write_path'] = Path(parameters['write_path']).absolute()
        
    if parameters.get('silent_mode') == None:
        parameters.update({'silent_mode' : False})
    else:
        parameters['silent_mode'] = bool(parameters['silent_mode'])
    
    list_arg = ['mode', 'image', 'video', 'images_path', 'video_path', 'model_name', 'path_classes', 'path_weights', 'path_config', 'confidence',
                  'nms_threshold', 'scale', 'size', 'mean', 'swapRB', 'groundtruth_path', 'write_path', 'silent_mode', 'adapter_name']

    entered_arg = parameters.keys()
    
    for arg in entered_arg:
        if not (arg in list_arg):
            raise ValueError(f'Incorrect parameter entered: {arg}')

    return parameters