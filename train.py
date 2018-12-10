from src.data.datagen import LoadData, DataGen
from src.img2seq import Img2SeqModel

from src.utils.lr_schedule import LRSchedule
from src.utils.general import Config

data = LoadData('./data/', 'images/', 'formulas.final.lst', 'formula_image_1to1.lst')
train_set, val_set, test_set, vocab = data()

test_set = DataGen(test_set[0], test_set[1])

config = Config(['./configs/data_small.json', './configs/vocab_small.json', './configs/training_small.json',
                './configs/model.json'])
config.save('./results/small/')

n_batches_epoch = ((len(train_set) + config.batch_size - 1) //
                        config.batch_size)

lr_schedule = LRSchedule(lr_init=config.lr_init,
        start_decay=config.start_decay*n_batches_epoch,
        end_decay=config.end_decay*n_batches_epoch,
        end_warm=config.end_warm*n_batches_epoch,
        lr_warm=config.lr_warm,
        lr_min=config.lr_min)

model = Img2SeqModel(config, './results/small/', vocab)

model.build_train(config)
model.train(config, val_set, test_set, lr_schedule)