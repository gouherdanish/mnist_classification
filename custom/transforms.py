
class InvertIntensity:
    def __call__(self, image):
        # Assuming the image is a PyTorch tensor with values in the range [0, 1]
        return 1 - image
