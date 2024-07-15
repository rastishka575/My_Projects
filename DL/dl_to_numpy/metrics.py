class Accuracy():
    def __call__(self, pred, label):
        pred = pred.argmax(axis=1)
        label = label.argmax(axis=1)
        pred = pred[pred == label]
        return (len(pred) / len(label)) * 100
