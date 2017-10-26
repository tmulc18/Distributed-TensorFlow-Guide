## DOWNPOUR

Similar to Hogwild! expect that it uses Adagrad to update the local workers.  Additionally, there is a communication window which servers as a time buffer for updates to the parameter server (although the original paper set the communication window to one, which voided the need for this buffer).
