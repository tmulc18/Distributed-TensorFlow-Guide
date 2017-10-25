There are two ways to set up a session in a distributed setting:

	1. Monitored Training Session
	2. Supervisor Session

The Monitored Training Session is the best option because it can be used in for synchronous training.  The Supervisor Session was initially created to handle threads and can be used for some distributed training, but overall offers less than the Monitored Training Session.  The schema for this is directory is as follows

* `dist_setup.py` -- python code for Monitored Training Session
* `dist_setup_sup.py` -- python code for Supervisor Session
* `run.sh` -- bash script for Monitored Training Session
* `run_sup.sh` -- bash script for Supervisor Session