from futscml_exports import Dotdict

TRANSFER_CONFIG = Dotdict()

# Training args.
# Expected directory structure:
# `data_root`/
#   pairs/
#     input/*.*
#     output/*.*
#   train/
#     input/*.*
TRANSFER_CONFIG.data_root = 'data/Kuba'
TRANSFER_CONFIG.logdir = 'Output_re'
TRANSFER_CONFIG.log_suffix = 'Kuba'
TRANSFER_CONFIG.device = 'cuda:0'
TRANSFER_CONFIG.resize = None
TRANSFER_CONFIG.video_fps = 25
TRANSFER_CONFIG.batch_size = 1
TRANSFER_CONFIG.lr = 1e-4 # 1e-2  # RMSprop works at 2e-4

# Training args.
TRANSFER_CONFIG.iter_limit = 1_000_000
TRANSFER_CONFIG.time_limit_minutes = 24 * 60
TRANSFER_CONFIG.log_image_update_every = 500
TRANSFER_CONFIG.log_video_update_every = 1000
TRANSFER_CONFIG.model_snapshot_hours = [0.5, 1, 2, 3, 6, 12]
TRANSFER_CONFIG.jit_model = False
TRANSFER_CONFIG.max_mem_for_log_video = 500

# Model args.
TRANSFER_CONFIG.model_config = Dotdict()
TRANSFER_CONFIG.model_config.width = 1.0
TRANSFER_CONFIG.model_config.resnet_blocks = 9
TRANSFER_CONFIG.model_config.tanh = True
TRANSFER_CONFIG.model_config.use_bias = True
TRANSFER_CONFIG.model_config.input_channels = 3
TRANSFER_CONFIG.model_config.output_channels = 3
TRANSFER_CONFIG.model_config.append_blocks = True
TRANSFER_CONFIG.model_config.norm_layer = 'instance_norm'

# Loss args.
TRANSFER_CONFIG.loss_args = Dotdict()
TRANSFER_CONFIG.loss_args.similarity_weight = 100.0
TRANSFER_CONFIG.loss_args.image_weight = 1.0
TRANSFER_CONFIG.loss_args.probed_multiframe_loss = False
TRANSFER_CONFIG.loss_args.layers = [1, 3, 6, 8, 11, 13, 15, 17,
                                    20, 22, 24, 26, 29, 31, 33, 35]
# Pixel loss args.
TRANSFER_CONFIG.loss_args.use_annealed_image_loss = False
TRANSFER_CONFIG.loss_args.annealed_image_loss_start = 5.0
TRANSFER_CONFIG.loss_args.annealed_image_loss_end = 0.0
TRANSFER_CONFIG.loss_args.annealed_image_loss_steps = 20_000

# Augmentations.
TRANSFER_CONFIG.augment_config = Dotdict()
TRANSFER_CONFIG.augment_config.disable_all = False
TRANSFER_CONFIG.augment_config.color_augments = Dotdict()
TRANSFER_CONFIG.augment_config.color_augments.adjust_contrast_min = 0.7
TRANSFER_CONFIG.augment_config.color_augments.adjust_contrast_max = 1.3
TRANSFER_CONFIG.augment_config.color_augments.adjust_hue_min = -0.1
TRANSFER_CONFIG.augment_config.color_augments.adjust_hue_max = 0.1
TRANSFER_CONFIG.augment_config.color_augments.adjust_saturation_min = 0.7
TRANSFER_CONFIG.augment_config.color_augments.adjust_saturation_max = 1.3

TRANSFER_CONFIG.augment_config.shape_augments = Dotdict()
TRANSFER_CONFIG.augment_config.shape_augments.angle_min = -15
TRANSFER_CONFIG.augment_config.shape_augments.angle_max = 15
TRANSFER_CONFIG.augment_config.shape_augments.hflip_chance = 0.5
TRANSFER_CONFIG.augment_config.shape_augments.tps_chance = 0.5
TRANSFER_CONFIG.augment_config.shape_augments.tps_scale = 0.075
