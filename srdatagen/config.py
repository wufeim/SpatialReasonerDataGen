from srdatagen.dnnlib import EasyDict

cfg = EasyDict()

cfg.resize_height = 640

# RAM configs
ram_cfg = EasyDict()
cfg.ram = ram_cfg
ram_cfg.pretrained_ckpt = 'pretrained_weights/ram_plus_swin_large_14m.pth'
ram_cfg.ignore_classes = remove_classes = [
    "room", "kitchen", "office", "house", "home", "building", "corner",
    "shadow", "carpet", "photo", "sea", "shade", "stall", "space", "aquarium",
    "apartment", "image", "city", "blue", "skylight", "hallway", "bureau",
    "modern", "salon", "doorway", "wall lamp", "scene", "sun", "sky", "smile",
    "cloudy", "comfort", "white", "black", "red", "green", "blue", "yellow",
    "purple", "pink", "stand", "wear", "area", "shine", "lay", "walk", "lead",
    "bite", "sing"]

# GroundingDINO configs
gdino_cfg = EasyDict()
cfg.gdino = gdino_cfg
gdino_cfg.model_config_path = 'Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
gdino_cfg.model_checkpoint_path = 'pretrained_weights/groundingdino_swint_ogc.pth'
gdino_cfg.box_threshold = 0.25
gdino_cfg.text_threshold = 0.2
gdino_cfg.nms_threshold = 0.5

# SAM configs
sam_cfg = EasyDict()
cfg.sam = sam_cfg
sam_cfg.cfg_path = 'configs/sam2.1/sam2.1_hq_hiera_l.yaml'
sam_cfg.ckpt_path = 'pretrained_weights/sam2.1_hq_hiera_large.pt'

# Filtering configs
filter_cfg = EasyDict()
cfg.filter = filter_cfg
filter_cfg.min_mask_area_ratio = 0.0005
filter_cfg.max_mask_area_ratio = 0.8
filter_cfg.mask_confidence_threshold = 0.3

# Reconstruct3D configs
r3d_cfg = EasyDict()
cfg.reconstruct3d = r3d_cfg
r3d_cfg.perspective_fields_model_name = 'Paramnet-360Cities-edina-centered'
r3d_cfg.wild_camera_model = 'ShngJZ/WildCamera'
r3d_cfg.dav2_backbone = 'vitl'
r3d_cfg.dav2_max_depth = 80
r3d_cfg.dav2_ckpt_path = 'pretrained_weights/paramnet_360cities_edina_rpf.pth'
r3d_cfg.min_points_threshold = 16
r3d_cfg.min_points_threshold_dbscan = 10
r3d_cfg.dbscan_eps = 0.2
r3d_cfg.dbscan_min_points = 10

# Pose configs
pose_cfg = EasyDict()
cfg.pose = pose_cfg
pose_cfg.padding = 1.2
pose_cfg.class_names = []  # will be filled in at the bottom
pose_cfg.multi_bin = EasyDict()
pose_cfg.multi_bin.num_bins = 40
pose_cfg.multi_bin.min_value = 0.0
pose_cfg.multi_bin.max_value = 6.2831853071795865
pose_cfg.multi_bin.border_type = 'periodic'
pose_cfg.backbone = 'facebook/dinov2-small'
pose_cfg.heads = [40, 40, 40]
pose_cfg.ckpt = 'pretrained_weights/pose_model_100.pth'

# Orient Anything configs
orientanything_cfg = EasyDict()
cfg.orientanything = orientanything_cfg
orientanything_cfg.ckpt_path = 'pretrained_weights/dino_weight.pt'
orientanything_cfg.dino_mode = 'large'
orientanything_cfg.in_dim = 1024
orientanything_cfg.out_dim = 360+180+180+2
orientanything_cfg.backbone = 'facebook/dinov2-large'

# Pose configs -- class_names
pose_cfg.class_names = [
    'person', 'car', 'chair', 'man', 'table', 'girl', 'woman', 'boy',
    'child', 'boat', 'computer', 'bird', 'bicycle', 'microphone',
    'face', 'armchair', 'animal', 'dog', 'speaker', 'baseball player',
    'motorbike', 'baby', 'baseball hat', 'plane', 'boot', 'clock', 'laptop',
    'horse', 'doll', 'fish', 'hat', 'cat', 'beak', 'drummer', 'gun',
    'sandal', 'smartphone', 'taxi', 'bench', 'bus', 'passenger train',
    'glasses', 'biker', 'cabinet', 'businessman', 'bookshelf', 'dancer',
    'construction worker', 'cart', 'couch', 'duck', 'shoe', 'projector',
    'bed', 'fork', 'city bus', 'camera', 'coach', 'park bench', 'cowboy',
    'rowboat', 'actor', 'seat', 'guitarist', 'bride', 'hiker', 'trailer truck',
    'train', 'automobile model', 'motorcycle', 'adult', 'video camera',
    'sheep', 'teddy', 'commander', 'beach chair', 'sedan', 'pigeon',
    'sports car', 'customer', 'barge', 'bull', 'bartender', 'minivan',
    'penguin', 'truck', 'eagle', 'dj', 'dirt bike', 'airliner', 'jeep',
    'tank', 'sailboat', 'college student', 'ferry', 'computer screen',
    'airman', 'mother', 'mountain bike', 'chef', 'commuter', 'excavator',
    'robot', 'cruise ship', 'van', 'fireplace', 'barbie', 'skull',
    'helicopter', 'suv', 'goose', 'artist', 'canoe', 'baby elephant',
    'father', 'skateboarder', 'student', 'golfer', 'referee', 'elephant',
    'astronaut', 'vehicle', 'closet', 'fireman', 'crown', 'spoon',
    'billboard', 'fire truck', 'sea lion', 'pony', 'tow truck', 'school bus',
    'athlete', 'goat', 'tractor', 'squirrel', 'singer', 'food truck',
    'wheelchair', 'nightstand'
]
