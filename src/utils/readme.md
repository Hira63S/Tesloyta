## trainer.py
### The main function that trains the model:

We define the parameters to be passed into the model like the model structure, the optimizer, lr_scheduler and other arguments related to them.

To run one epoch, we pass in phase, epoch and dataloader:
- Phase: which mode is the model in train, eval etc
- epoch: which epoch
- Dataloader: dataloader .. duh

We start the timing and set the model modes i.e train or eval.
We then import the metric logger function that only has four variables: val, avg, sum and count.
We initiate the metric logger object for loss, class_labels, score_loss and bbox_loss. There is also an update function associated with it so that we can update whenever an epoch ends.

Then we enumerate the data_loader, checking if the iter_id is less than the num_iters we passed in, the loop breaks.
Then we check if an item in the batch does not have image_meta i.e. all the image metrics like id, mean, orig size, we pass it to the device. i.e. for the model to train.
we update the data_timer sincei t is a metric_loggers object and the class has update function that updates those 4 values.
Then we get the loss, and loss_stats from the model class.
loss = loss.mean()

We set the optimizer to zero_grad() and propagate the loss backward.
We clip the grad norm and do the step with the optimizer.
Then we print the output of the first epoch.
We update the loss stat in metric logger related to loss
and then update the time.

We also define the lr_scheduler if the mode is train and it update the status for all the metric_loggers.
Then we define the train_epoch as well as the val_epoch function
#
### Next is setting device which would be gpu if you have access to one, otherwise it would be a cpu.
