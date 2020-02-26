# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'scales': [[1.0], [1.0], [1.0]],
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
    'anchor_num': 2,
    'fpn_num': 3
}

cfg_re50_5fpn = {
    'name': 'Resnet50',
    'min_sizes': [[16, 20.16, 25.40], [32, 40.32, 50.80], [64, 80.63, 101.59], [128, 161.26, 203.19], [256, 322.54, 406.37]],
    'steps': [4, 8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'scales': [[1.0], [1.0], [1.0], [1.0], [1.0]],
    'gpu_train': True,
    'pretrain': True,
    'return_layers': {'layer1': 1, 'layer2': 2, 'layer3': 3, 'layer4':4},
    'in_channel': 256,
    'out_channel': 256,
    'anchor_num': 3,
    'fpn_num': 5
}

cfg_re50_3fpn = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'scales': [[1.0], [1.0], [1.0]],
    'gpu_train': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256,
    'anchor_num': 2,
    'fpn_num': 3
}

cfg_efcb0_5fpn = {
    'name': 'efficientnet-b0',
    'min_sizes': [[16, 20.16, 25.40], [32, 40.32, 50.80], [64, 80.63, 101.59], [128, 161.26, 203.19], [256, 322.54, 406.37]],
    'steps': [4, 8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'scales': [[1.0], [1.0], [1.0], [1.0], [1.0]],
    'return_layers_idcs': [2, 4, 10, 15],
    'in_channel_list': [24, 40, 112, 320],
    'out_channel': 256,
    'anchor_num': 3,
    'fpn_num': 5
}

cfg_efcb0_3fpn = {
    'name': 'efficientnet-b0',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'scales': [[1.0], [1.0, 1.41], [1.0]],
    'return_layers_idcs': [4, 10, 15],
    'in_channel_list': [40, 112, 320],
    'out_channel': 256,
    'anchor_num': 2,
    'fpn_num': 3
}

cfg_efcb1_3fpn = {
    'name': 'efficientnet-b1',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'scales': [[1.0], [1.0, 1.41], [1.0]],
    'return_layers_idcs': [7, 15, 22],
    'in_channel_list': [40, 112, 320],
    'out_channel': 256,
    'anchor_num': 2,
    'fpn_num': 3
}

cfg_efcb2_3fpn = {
    'name': 'efficientnet-b2',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'scales': [[1.0], [1.0, 1.41], [1.0]],
    'return_layers_idcs': [7, 15, 22],
    'in_channel_list': [48, 120, 352],
    'out_channel': 256,
    'anchor_num': 2,
    'fpn_num': 3
}

cfg_efcb3_3fpn = {
    'name': 'efficientnet-b3',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'scales': [[1.0], [1.0, 1.41], [1.0]],
    'return_layers_idcs': [7, 17, 25],
    'in_channel_list': [48, 136, 384],
    'out_channel': 256,
    'anchor_num': 2,
    'fpn_num': 3
}

cfg_efcb4_3fpn = {
    'name': 'efficientnet-b4',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'scales': [[1.0], [1.0, 1.41], [1.0]],
    'return_layers_idcs': [9, 21, 31],
    'in_channel_list': [56, 160, 448],
    'out_channel': 256,
    'anchor_num': 2,
    'fpn_num': 3
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

cfg_extd = {
    'name': 'extd',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 128,
    'out_channel': 128
}