from src.data.datagen import LoadData, DataGen
from src.img2seq import Img2SeqModel

from src.utils.lr_schedule import LRSchedule
from src.utils.general import Config
from src.evaluation.text import score_files

data = LoadData('./data/', 'images/', 'formulas.final.lst', 'formula_image_1to1.lst')
train_set, val_set, test_set, vocab = data()

test_set = DataGen(test_set[0], test_set[1])
#val_set = DataGen(val_set[0], val_set[1])

dir_output = './results/large/'

config_data  = Config(dir_output + "data.json")
config_vocab = Config(dir_output + "vocab.json")
config_model = Config(dir_output + "model.json")

#vocab = Vocab(config_vocab)
model = Img2SeqModel(config_model, dir_output, vocab)
model.build_pred()
model.restore_session(dir_output + "model.weights/")

config_eval = Config({"dir_answers": dir_output + "formulas_test/",
                          "batch_size": 20})

files, perplexity = model.write_prediction(config_eval, test_set)
formula_ref, formula_hyp = files[0], files[1]

# score the ref and prediction files
scores = score_files(formula_ref, formula_hyp)
scores["perplexity"] = perplexity
msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in scores.items()])
model.logger.info("- Test Txt: {}".format(msg))


