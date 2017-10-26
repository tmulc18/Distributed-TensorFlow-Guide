## Multiple GPUs Single Machine

Use environment variables to manually override the available GPUs in a TensorFlow process.  There is a way to do this without using environment variables, but it's a not worth the effort (if you really need this, you can remap the available devices so the GPU you want to use is labeled as device 0, then set visible devices to 0).
