from models import LSTM_model
model = LSTM_model(0, num_classes=10, img_width=32, img_height=32, img_channels=3,l2_reg=0,batch_norm = True)

print(model.summary())