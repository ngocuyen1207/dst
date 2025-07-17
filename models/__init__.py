class BaseDST:
    """Abstract base class for DST models."""
    def reset(self):
        """Reset belief state at the start of a dialogue."""
        raise NotImplementedError

    def update(self, user_utterance):
        """Update belief state based on user utterance."""
        raise NotImplementedError
