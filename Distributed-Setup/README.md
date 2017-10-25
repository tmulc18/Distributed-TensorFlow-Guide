There are many ways to set up a session in a distributed setting but we demonstrate two in this example:

	1. Monitored Training Session
	2. Supervisor Session

The Monitored Training Session is the best option because it can handle many hooks and can be used for synchronous training.  The Supervisor Session offers suppport for to handling threads and can be used for some distributed training, but overall offers less than the Monitored Training Session.  The schema for this is directory is as follows

* `dist_setup.py` -- python code for Monitored Training Session
* `dist_setup_sup.py` -- python code for Supervisor Session
* `run.sh` -- bash script for Monitored Training Session
* `run_sup.sh` -- bash script for Supervisor Session