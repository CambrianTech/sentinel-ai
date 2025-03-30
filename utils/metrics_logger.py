# utils/metrics_logger.py

import logging


class MetricsLogger:
    def __init__(self, log_file="training.log"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler and set level to info
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create console handler with a higher level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_metrics(self, epoch, step, train_loss, val_loss, perplexity, active_head_count, param_count):
        log_message = (
            f"Epoch: {epoch}, Step: {step}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Perplexity: {perplexity:.4f}, Active Heads: {active_head_count}, "
            f"Parameter Count: {param_count}"
        )
        self.logger.info(log_message)

    def log_eval(self, val_loss, perplexity, baseline_ppl):
        log_message = (
            f"Validation Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}, "
            f"Baseline Perplexity: {baseline_ppl:.4f}"
        )
        self.logger.info(log_message)

    def log_train(self, train_loss):
        log_message = f"Training Loss: {train_loss:.4f}"
        self.logger.info(log_message)

    def log_active_heads(self, active_head_count):
        log_message = f"Active Heads: {active_head_count}"
        self.logger.info(log_message)
