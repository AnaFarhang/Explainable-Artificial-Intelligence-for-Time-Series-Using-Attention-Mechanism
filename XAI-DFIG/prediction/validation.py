import torch

class validation_loop_func:
    def __init__(self, dataloader, model, loss_fn, epoch):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        validation_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                pred, attnval, attnallheadslayers, wq, wk, wv, attn_raw  = model(X.float())
                validation_loss += loss_fn(pred, y).item()
                # This line calculates the number of correct predictions.
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        validation_loss /= num_batches
        self.validation_loss=validation_loss
        correct /= size
        self.accuracy=correct
        print(f"Accuracy: {(100 * correct):>8f}%, Avg loss: {validation_loss:>8f}")

