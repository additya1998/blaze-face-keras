import tensorflow as tf
from data import UplaraDataset
from network import blaze_face_detector
from loss import location_loss, confidence_loss
from tensorflow.keras.optimizers import Adam

train_data = UplaraDataset().get_data()
model = blaze_face_detector((128, 128, 3))
opt = Adam(lr=0.001, decay=0.01 / 100)

for epoch in range(25):
    print("Epoch:", epoch)
    epoch_conf_loss, epoch_loc_loss = 0, 0
    for (images, labels, boxes) in train_data:
        with tf.GradientTape() as tape:
            output_scores, output_boxes = model(images)
            loc_loss = location_loss(boxes, labels, output_boxes, output_scores)
            conf_loss = confidence_loss(labels, output_scores)
            total_loss = loc_loss + conf_loss
            epoch_conf_loss += conf_loss.numpy()
            epoch_loc_loss += loc_loss.numpy()
        grads = tape.gradient(total_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
    print('Epoch Confidence loss:', epoch_conf_loss)
    print('Epoch Location loss:', epoch_loc_loss)
    print('')
