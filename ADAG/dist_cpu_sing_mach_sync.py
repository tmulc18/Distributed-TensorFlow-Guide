"""
Asynchronous Distributed Adaptive Gradients (ADAG)
Performs asynchronous updates with update window 

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
	cluster_spec = {'ps':['localhost:2222'],
					'worker':['localhost:2223','localhost:2224']}
	n_pss = len(cluster_spec['ps']) #the number of parameter servers
	n_workers = len(cluster_spec['worker']) #the number of worker nodes
	cluster = tf.train.ClusterSpec(cluster_spec) #allows this node know about all other nodes

	if FLAGS.job_name == 'ps': #checks if parameter server
		server = tf.train.Server(cluster,job_name="ps",task_index=FLAGS.task_index,config=config)
		server.join()
	else: #it must be a worker server
		is_chief = (FLAGS.task_index == 0) #checks if this is the chief node
		server = tf.train.Server(cluster,job_name="worker",task_index=FLAGS.task_index,config=config)
		
		# Graph
		with tf.device(tf.train.replica_device_setter(ps_tasks=n_pss\
                ,worker_device="/job:%s/task:%d/cpu:0" % (FLAGS.job_name,FLAGS.task_index))):

			a = tf.Variable(tf.constant(0.,shape=[2]),dtype=tf.float32)
			b = tf.Variable(tf.constant(0.,shape=[2]),dtype=tf.float32)
			c=a+b

			global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
			target = tf.constant(100.,shape=[2],dtype=tf.float32)
			loss = tf.reduce_mean(tf.square(c-target))

			# all workers use the same learning rate and it is decided on by the task 0 
			# or maybe the from the graph of the chief worker
			base_lr = .0001
			optimizer = tf.train.GradientDescentOptimizer(base_lr) #the learning rate set here is global

			#local optimizers
			optimizers=[]
			local_steps = []
			for w in range(n_workers):
				local_steps.append(tf.Variable(0,dtype=tf.int32,trainable=False,name='local_step_%d'%w))
				optimizers.append(tf.train.GradientDescentOptimizer(base_lr))
		
			# ADAG (simplest case since all batches are the same)
			update_window = 5 # T: update window, a.k.a number of gradients to use before sending to ps
			grad_list = [] # the array to store the gradients through the communication window
			for t in range(update_window):
				if t != 0:
					with tf.control_dependencies([opt_local]):
						grads, varss = zip(*optimizers[FLAGS.task_index].compute_gradients(loss)) #compute gradients using local optimizer
				else:
					grads, varss = zip(*optimizers[FLAGS.task_index].compute_gradients(loss)) #compute gradients using local optimizer
				grad_list.append(grads) #add gradients to the list
				opt_local = optimizers[FLAGS.task_index].apply_gradients(zip(grads,varss),
										global_step=local_steps[FLAGS.task_index]) #update local parameters
			grads = tf.reduce_mean(grad_list,axis=0) #taking the mean is the same as dividing the sum of gradients by T
			grads = tuple([grads[i]for i in range(len(varss))])
			opt = optimizer.apply_gradients(zip(grads,varss),global_step=global_step) #apply the gradients globally

			# Init op
			init = tf.global_variables_initializer() # must come after other init ops


		# Session
		stop_hook = tf.train.StopAtStepHook(last_step=20)
		hooks = [stop_hook]
		scaff = tf.train.Scaffold(init_op=init)

		#Monitored Training Session
		sess = tf.train.MonitoredTrainingSession(master = server.target,is_chief=is_chief,config=config,
													scaffold=scaff,hooks=hooks,stop_grace_period_secs=10)

		if is_chief:
			time.sleep(5) #grace period to wait on other workers before starting training

		# Train until hook stops session
		print('Starting training on worker %d'%FLAGS.task_index)
		while not sess.should_stop():
			_,r,gs,ls = sess.run([opt,c,global_step,local_steps[FLAGS.task_index]])

			print(r,"global step: "+str(gs),"worker: "+str(FLAGS.task_index),"local step: "+str(ls))

			if gs % 7 == 1:
				for j in grad_list:
					print(sess.run(j),FLAGS.task_index)

			# if is_chief: time.sleep(1)
			time.sleep(1)
		print('Done',FLAGS.task_index)

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
