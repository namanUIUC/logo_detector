" main "
import sys
from models import BOG

base_path = "./"
data_path = "data/"
train_img = ["CapitalOne.png", "BankOfAmerica.jpg",
             "JPMorganChase.png", "Citigroup.jpg", "WellsFargo.jpg"]

train_labels = ["CapitalOne", "BOA", "Chase", "Citi", "WellsFargo"]

train_images = [base_path + data_path + s for s in train_img]

# test_img = ["BankOfAmerica_header.png", "JPMorganChase_header.png",
#             "Citigroup_header.png", "WellsFargo_header.png", "CapitalOne_header.png"]

# test_images = [base_path + data_path + s for s in test_img]


def model(image):

    model = BOG.Detector()
    model.train(train_images, train_labels)
    model.predict(image)


if __name__ == "__main__":
    model(sys.argv[1])
