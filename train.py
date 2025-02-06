from network_architecture import transfer_model, optimizer, train_data_loader, val_data_loader
import torch
import torch.nn as nn

def train(model, 
          optimizer, 
          loss_fn, 
          train_loader, 
          val_loader, 
          epochs=20, 
          device='cpu',
          save_path='models/trained_model.pth'):
    
    training_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    
    ## Unfreezing specific layers for Transformer, Unfreeze specific transformer encoder blocks
    unfreeze_layers = [
        model.blocks[6],  # Example: unfreeze the 6th block
        model.blocks[7],  # Example: unfreeze the 7th block
    ]

    ## Enable gradient updates for the selected layers
    print("Unfreezing Layers...")
    for layer in unfreeze_layers:
        for param in layer.parameters():
            param.requires_grad = True
    
    print("Starting Epochs...")
    best_val_loss = float('inf') # This initalizes a very high value
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {training_loss}")
        training_loss_list.append(training_loss)
        
        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item()
            preds = torch.max(nn.functional.softmax(output, dim=1), dim=1)[1]
            num_correct += (preds == targets).sum().item()
            num_examples += targets.size(0)
            
        valid_loss /= len(val_loader)
        accuracy = num_correct / num_examples
        print('Epoch: {}, Training Loss: {:.5f}, Validation Loss: {:.5f}, Accuracy = {:.5f}'.format(epoch+1, training_loss, valid_loss, accuracy))
        val_loss_list.append(valid_loss)
        val_accuracy_list.append(accuracy)
        
        # Save the model if validation loss improves
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model at Epoch {epoch+1} with Validation Loss: {valid_loss:.5f}")
    
    # Printing results
    print("---------- Results ----------")
    print("Training Loss List: ")
    print(training_loss_list)
    print("")
    print("Validation Loss List: ")
    print(val_loss_list)
    print("")
    print("Validation Accuracy List: ")
    print(val_accuracy_list)
            
train(transfer_model, 
      optimizer, 
      torch.nn.CrossEntropyLoss(),
      train_data_loader, 
      val_data_loader)