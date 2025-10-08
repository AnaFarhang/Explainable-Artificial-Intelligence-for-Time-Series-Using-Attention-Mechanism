


class train_loop_func:
    def __init__(self, dataloader, model, loss_fn, optimizer, epoch):
        # The total number of samples in the dataset associated with the given dataloader.
        size = len(dataloader.dataset)
        # This line calculates the total number of batches in the dataloader
        num_batches = len(dataloader)
        # This line initializes a variable train_loss to keep track of the accumulated training loss for the current epoch.
        train_loss = 0
        # This line sets the value for the log step, which determines how often the loss is printed during training.
        # In this case, the loss will be printed every 10 batches.
        log_step = 10
        # The following loop iterates over the batches in the dataloader
        for batch, (X, y) in enumerate(dataloader):
            # This line passes the input data X to the model to obtain the predicted outputs
            pred, attntra, attnallheadslayers, wq, wk, wv, attn_raw = model(X.float())
            # This line calculates the loss by passing the predicted outputs pred and the ground truth labels y.

            loss = loss_fn(pred, y)
            #  This line accumulates the loss for the current batch by adding it to the train_loss variable.
            train_loss += loss
            # This line clears the gradients of all model parameters before computing the gradients for the current batch.
            # It is necessary to prevent gradient accumulation.
            optimizer.zero_grad()
            # This line performs backpropagation to compute the gradients of the model parameters with respect to the loss.
            loss.backward()
            # This line updates the model's parameters by taking an optimizer step.
            # It adjusts the parameters based on the computed gradients and the optimizer's update rule.
            optimizer.step()
            # Print loss values
            if batch % log_step == 0:
                loss, current = loss.item(), batch * len(X)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        # It divides the accumulated loss by the number of batches
        train_loss /= num_batches
        self.train_loss=train_loss.item()
