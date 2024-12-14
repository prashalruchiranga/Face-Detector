from models.keras_ssd7 import build_model
from tensorflow.keras.optimizers import Adam
from keras_loss_function.keras_ssd_loss import SSDLoss

class FaceDetector:
    def __init__(self, img_height, img_width, weights_path):
        self.img_height = img_height
        self.img_width = img_width
        self.weights_path = weights_path
        self.model = self._create_model()

    def _create_model(self):
        self.ssd = build_model(
            image_size=(self.img_height, self.img_width, 3),
            n_classes=1,
            mode='training',
            l2_regularization=0.0005,
            min_scale=0.07,
            max_scale=0.9,
            aspect_ratios_global=[0.5, 1.0, 2.0],
            aspect_ratios_per_layer=None,
            two_boxes_for_ar1=True,
            steps=None,
            offsets=None,
            clip_boxes=False,
            variances=[1.0, 1.0, 1.0, 1.0],
            normalize_coords=True,
            subtract_mean=127.5,
            divide_by_stddev=127.5
            )
        
        if self.weights_path:
            self.ssd.load_weights(self.weights_path)
        adam = Adam(learning_rate=0.001)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.ssd.compile(optimizer=adam, loss=ssd_loss.compute_loss)
        return self.ssd

    def predict_bbxs(self, input_image):
        y_pred = self.model.predict(input_image)
        return y_pred
