def train(model, loss_function, optimizer, train_loader, val_loader, save_file_name, max_epochs_stop=3, n_epochs, print_every=2):

    # initiate early stopping:
    epochs_fit = 0
    min_val_loss = np.Inf
    max_val_acc = 0
    history = []

    # print statements:
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training'\n)

    overall_start = timer()

    for epoch in range(n_epochs):
        # initiate current accuracy and loss
        train_acc = 0
        train_loss = 0
        # set the eval mode
        model.train()
        start = timer()

        for i, (data, target) in enumerate(train_loader):
            # define the device first i.e. if GPU go to gpu else CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

        # zero the gradients:
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                loss = loss_function(outputs, labels)

            # backward
                loss.backward()
                optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(predictions == labels.data)

        total_loss = train_loss/len(train_loader.dataset) # dataset returns an object with dataset details
        total_acc = train_acc.double() / len(train_loader.dataset) # length is the total number of examples in the data_loader object

        print(f'Epoch: {epoch}\t{100 * (i + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.', end='\r')
        print('Train Loss: {:.4f}; Accuracy {:.4f}'.format(total_loss, total_acc))

    else:
        model.epochs += 1

        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()

            current_loss = 0.0
            current_acc = 0

            for data, target in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = loss_function(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)
                valid_acc += torch.sum(predictions == labels.data)

            total_val_loss = valid_loss / len(val_loader.dataset)
            total_val_acc = valid_acc.double()/ len(val_loader.dataset)

            history.append([train_loss, valid_loss, train_acc, valid_acc])

            if (epoch + 1) % print_every == 0:
                print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f}\tValidation Loss: {valid_loss:.4f}')
                print(f'\t\Training Accuracy: {100 * train_acc:.2f}%\t validation Accuracy: {100 * valid_acc:.2f}%')
                print('Test Loss: {:.4f}; Accuracy{:.4f}'.format(total_val_loss, total_val_acc))

            # when to save the model:

            if valid_loss < min_val_loss:

                torch.save(model.state_dict(), save_file_name)

                epochs_fit = 0
                min_val_loss = valid_loss
                max_val_acc = valid_acc
                best_epoch = epoch

            else:
                epochs_fit += 1
                if epochs_fit >= max_epochs_stop:
                    print(
                    f'\nEarly Stopping! Total Epochs: {epoch}. Best epoch: {best_epoch} with loss: {min_val_loss:.2f} and acc:{100 * valid_acc:.2f}%')
                    total_time = timer() - overall_start
                    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')

                    model.load_state_dict(torch.load(save_file_name))
                    # attach the optimizer
                    model.optimizer = optimizer

                    history = pd.DataFrame(history, columns = ['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

                    return model, history
