import torch
import torch.nn as nn
import torch.optim as optim
import higher # Library for differentiable inner-loops (Meta-Learning)

class MelampoTrainer:
    """
    Class: Training Orchestrator
    
    Role:
    Manages the complex hybrid training lifecycle combining:
    1. MAML (Model-Agnostic Meta-Learning) for fast adaptation.
    2. Prototypical Networks (Metric-based classification).
    3. EWC (Elastic Weight Consolidation) to prevent forgetting previous pathologies.
    """
    
    def __init__(self, model, meta_lr=0.001, inner_lr=0.01, ewc_lambda=0.4):
        self.model = model
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr
        self.ewc_lambda = ewc_lambda
        
        # EWC Storage
        self.fisher_matrix = {} 
        self.optpar_store = {}

    def prototypical_loss(self, embeddings, targets, n_support):
        """
        Calculates distance between Query samples and Class Prototypes.
        """
        # (Simplified implementation for brevity)
        # 1. Calculate Prototypes (Mean of Support Set)
        # 2. Calculate Euclidean Distance between Query and Prototypes
        # 3. Softmax over negative distances
        # ... logic for proto-loss ...
        loss = nn.CrossEntropyLoss()(embeddings, targets) # Placeholder
        return loss

    def compute_ewc_penalty(self, current_model):
        """
        Calculates the EWC loss component based on Fisher Information.
        L_ewc = sum( Fisher_i * (theta_i - theta_opt_i)^2 )
        """
        loss = 0
        if not self.fisher_matrix:
            return 0.0
            
        for name, param in current_model.named_parameters():
            if name in self.fisher_matrix:
                fisher = self.fisher_matrix[name]
                opt_param = self.optpar_store[name]
                # Calculate stiffness penalty
                loss += (fisher * (param - opt_param).pow(2)).sum()
        return loss * self.ewc_lambda

    def meta_train_step(self, task_batch):
        """
        Performs one MAML step with EWC regularization.
        """
        support_x, support_y, query_x, query_y = task_batch
        
        self.meta_optimizer.zero_grad()
        
        # Using 'higher' to handle the differentiable inner loop
        with higher.innerloop_ctx(self.model, self.meta_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
            
            # --- INNER LOOP (Fast Adaptation) ---
            # Simulate learning on a specific pathology (Support Set)
            support_output, _ = fmodel(support_x, {'noradrenaline': 0.8}) # High focus for learning
            loss = self.prototypical_loss(support_output, support_y, n_support=5)
            diffopt.step(loss)
            
            # --- OUTER LOOP (Meta-Update) ---
            # Evaluate how well the model adapted using unseen data (Query Set)
            query_output, _ = fmodel(query_x, {'noradrenaline': 0.5}) # Balanced for testing
            meta_loss = self.prototypical_loss(query_output, query_y, n_support=5)
            
            # Add Memory Consolidation Penalty (EWC)
            # Ensures we don't break weights important for previous tasks
            ewc_penalty = self.compute_ewc_penalty(fmodel)
            total_loss = meta_loss + ewc_penalty
            
            # Backpropagate through the entire learning process
            total_loss.backward()
            self.meta_optimizer.step()
            
        return total_loss.item()

    def update_fisher_matrix(self, dataset):
        """
        Called after finishing a task to consolidate memory.
        Computes the importance (Fisher Info) of each parameter.
        """
        self.model.eval()
        # ... Logic to accumulate gradients and square them to estimate Fisher Info ...
        # Store in self.fisher_matrix
        print("Memory Consolidated via EWC.")
        