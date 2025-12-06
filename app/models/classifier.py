from PIL import Image


class NSFWClassifier:
   
    # Dummy classifier: always says 'possible violation'.
    # Replace with real NSFW/safety model later.

    def __init__(self):
        pass

    def predict(self, image: Image.Image):
        return {
            "possible_violation": True,
            "categories": ["nudity_explicit"],
        }
