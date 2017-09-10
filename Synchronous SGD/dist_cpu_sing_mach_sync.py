"""
Synchronous SGD

Author: Tommy Mulc
"""

from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os
FLAGS = None
log_dir = '/logdir'

def main():
	# Configure
	config=tf.ConfigProto(log_device_placement=False)

	#Distributed Baggage
	cluster = tf.train.ClusterSpec({'ps':['localhost:2222'],
										'worker':['localhost:2223','localhost:2224']}) #allows this node know about all other nodes
	if FLAGS.job_name == 'ps': #checks if parameter server
		server = tf.train.Server(cluster,job_name="ps",task_index=FLAGS.task_index,config=config)
		server.join()
	else: #it must be a worker server
		is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
		server = tf.train.Server(cluster,job_name="worker",task_index=FLAGS.task_index,config=config)
		
		# Graph
		with tf.device(tf.train.replica_device_setter(ps_tasks=1\
                ,worker_device="/job:%s/task:%d/cpu:0" % (FLAGS.job_name,FLAGS.task_index))):

			a = tf.Variable(tf.constant(0.,shape=[2]),dtype=tf.float32)
			b = tf.Variable(tf.constant(0.,shape=[2]),dtype=tf.float32)
			c=a+b

			global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
			target = tf.constant(100.,shape=[2],dtype=tf.float32)
			loss = tf.reduce_mean(tf.square(c-target))

			# create an optimizer then wrap it with SynceReplicasOptimizer
			optimizer = tf.train.GradientDescentOptimizer(.0001) #the learning rate set here is global
			optimizer1 = tf.train.SyncReplicasOptimizer(optimizer,replicas_to_aggregate=2,total_num_replicas=2)
			
			opt = optimizer1.minimize(loss,global_step=global_step) # averages gradients

		# Session
		sync_replicas_hook = optimizer1.make_session_run_hook(is_chief)
		stop_hook = tf.train.StopAtStepHook(last_step=10)
		hooks = [sync_replicas_hook,stop_hook]

		#Monitored Training Session
		sess = tf.train.MonitoredTrainingSession(master = server.target, 
                                             is_chief=is_chief,
                                             config=config,
                                             hooks=hooks,
                                             stop_grace_period_secs=10)

		print('Starting training on worker %d'%FLAGS.task_index)
		while not sess.should_stop():
			# if sess.should_stop(): print("should stopped"); break
			_,r,gs=sess.run([opt,c,global_step])

			print(r,gs,FLAGS.task_index)

			if is_chief: print(r,gs,FLAGS.task_index); time.sleep(1)
			time.sleep(1)
		print('Done',FLAGS.task_index)

		# if is_chief:
		# 	sess.run()
		# Must stop threads first
		time.sleep(10) #grace period to wait before closing session
		sess.close()
		print('Session from worker %d closed cleanly'%FLAGS.task_index)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Flags for defining the tf.train.ClusterSpec
	parser.add_argument(
    	"--job_name",
    	type=str,
    	default="",
    	help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
	parser.add_argument(
    	"--task_index",
    	type=int,
    	default=0,
    	help="Index of task within the job"
    )
	FLAGS, unparsed = parser.parse_known_args()
	print(FLAGS.task_index)
	main()
