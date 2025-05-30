import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os

class AlphaZeroLiteTrainer:
    """Trainer for the AlphaZeroLite model"""
    def __init__(self, model, batch_size=32, weight_decay=1e-4, lr=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.network.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.device = model.device
        
        # For LR scheduling
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train_batch(self, states, policies, values):
        """Train on a batch of data"""
        # Move data to device
        states = states.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)
        
        # Forward pass
        policy_logits, value_pred = self.model.network(states)
        
        # Compute losses
        policy_loss = -torch.sum(policies * F.log_softmax(policy_logits, dim=1)) / policies.size(0)
        value_loss = F.mse_loss(value_pred.squeeze(-1), values)
        
        # Weight the losses (can be tuned)
        total_loss = policy_loss + value_loss
        
        # Backward pass and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, dataset, num_epochs=1, checkpoint_dir=None, checkpoint_freq=1):
        """Train for multiple epochs"""
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        losses = []
        self.model.network.train()
        
        for epoch in range(num_epochs):
            epoch_losses = []
            start_time = time.time()
            
            for batch_idx, (states, policies, values) in enumerate(dataloader):
                loss = self.train_batch(states, policies, values)
                epoch_losses.append(loss)
                
                # Print progress every 100 batches
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = {k: np.mean([loss[k] for loss in epoch_losses[-10:]]) for k in epoch_losses[0]}
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                         f"Policy Loss: {avg_loss['policy_loss']:.4f}, "
                         f"Value Loss: {avg_loss['value_loss']:.4f}")
            
            # Calculate average loss for the epoch
            avg_epoch_loss = {k: np.mean([loss[k] for loss in epoch_losses]) for k in epoch_losses[0]}
            losses.append(avg_epoch_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} complete, "
                 f"Policy Loss: {avg_epoch_loss['policy_loss']:.4f}, "
                 f"Value Loss: {avg_epoch_loss['value_loss']:.4f}, "
                 f"Time: {time.time() - start_time:.2f}s")
            
            # Step the scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_freq == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
                self.model.save_model(checkpoint_path)
        
        return losses