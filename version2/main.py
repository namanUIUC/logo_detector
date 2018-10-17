" main "
from models import BOG

base_path = "./"
data_path = "data/"
train_img = [
            "CapitalOne.png",\
            "SquareCash.png",\
            "BankOfAmerica.jpg",\
            "JPMorganChase.png",\
            "Citigroup.jpg",\
            "Robinhood.png",\
            "WellsFargo.jpg"
            ]

test_img =  [
            "BankOfAmerica_header.png",\
            "JPMorganChase_header.png",\
            "Citigroup_header.png",\
            "WellsFargo_header.png",\
            "CapitalOne_header.png",\
            ]

train_images = [base_path+data_path+s for s in train_img]
test_images  = [base_path+data_path+s for s in test_img]

train_labels= [
                "CapitalOne",
                "others1",
                "BOA",
                "Chase",
                "Citi",
                "other2",
                "WellsFargo"
              ]

model = BOG.Detector()
model.train(train_images, train_labels)
model.predict(test_images[4]) 
