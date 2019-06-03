from text import symbols


class hparams:
	################################
	# Data Parameters              #
	################################
	max_input = 140 # drop long text input
	text_cleaners=['english_cleaners']

	################################
	# Audio                        #
	################################
	num_mels = 80
	num_freq = 1025
	sample_rate = 16000
	frame_length_ms = 50
	frame_shift_ms = 12.5
	preemphasis = 0.97
	min_level_db = -100
	ref_level_db = 20
	power = 1.5
	gl_iters = 100

	################################
	# Model Parameters             #
	################################
	n_symbols = len(symbols)
	symbols_embedding_dim = 256 # 512

	# Encoder parameters
	encoder_kernel_size = 5
	encoder_n_convolutions = 3
	encoder_embedding_dim = 256 # 512

	# Decoder parameters
	n_frames_per_step = 1
	decoder_rnn_dim = 512 # 1024
	prenet_dim = 128 # 256
	max_decoder_steps = 1000
	gate_threshold = 0.5
	p_attention_dropout = 0.1
	p_decoder_dropout = 0.1

	# Attention parameters
	attention_rnn_dim = 512 # 1024
	attention_dim = 256 # 128

	# Location Layer parameters
	attention_location_n_filters = 32
	attention_location_kernel_size = 31

	# Mel-post processing network parameters
	postnet_embedding_dim = 256 # 512
	postnet_kernel_size = 5
	postnet_n_convolutions = 5

	################################
	# Train                        #
	################################
	is_cuda = True
	pin_mem = True
	n_workers = 8
	lr = 1e-3
	wu = False
	wu_step = 4000
	max_iter = 2e5
	batch_size = 32
	iters_per_log = 50
	iters_per_sample = 100
	iters_per_ckpt = 1000
	weight_decay = 1e-6
	grad_clip_thresh = 1.0
	mask_padding = True
	eg_text = 'Tacotron is awesome!'
