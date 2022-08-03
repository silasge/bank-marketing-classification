all: download unzip train_test_split train

# Data URL
BANK_MARKETING_URL = https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip

# Diretórios 
EXTERNAL_DATA_PATH = ./data/external
RAW_DATA_PATH = ./data/raw
INTERIM_PATH = ./data/interim
REFERENCES_PATH = ./references
MODELS_PATH = ./models
SRC_PATH = ./bank_marketing_classification

# Poetry run
POETRY_RUN = poetry run python

# Parâmetros
TEST_SIZE = 0.25
RANDOM_STATE = 42
CV = 5
N_ITER = 50
SCORING = roc_auc

# Modelos

MODELS = $(MODELS_PATH)/lr.pkl $(MODELS_PATH)/svc.pkl $(MODELS_PATH)/dt.pkl $(MODELS_PATH)/rf.pkl

download: $(EXTERNAL_DATA_PATH)/bank-additional.zip
unzip: $(RAW_DATA_PATH)/bank-additional-full.csv $(REFERENCES_PATH)/bank-additional-names.txt
train: $(MODELS)
	
$(EXTERNAL_DATA_PATH)/bank-additional.zip: $(SRC_PATH)/download_data.py
	$(POETRY_RUN) $< $(BANK_MARKETING_URL) $(EXTERNAL_DATA_PATH)

$(RAW_DATA_PATH)/bank-additional-full.csv: $(SRC_PATH)/unzip_data.py $(EXTERNAL_DATA_PATH)/bank-additional.zip
	$(POETRY_RUN) $< $(word 2, $^) bank-additional/bank-additional-full.csv $(RAW_DATA_PATH)

$(REFERENCES_PATH)/bank-additional-names.txt: $(SRC_PATH)/unzip_data.py $(EXTERNAL_DATA_PATH)/bank-additional.zip
	$(POETRY_RUN) $< $(word 2, $^) bank-additional/bank-additional-names.txt $(REFERENCES_PATH)

train_test_split: $(SRC_PATH)/split_data.py $(RAW_DATA_PATH)/bank-additional-full.csv
	$(POETRY_RUN) $< $(word 2, $^) \
	                 $(INTERIM_PATH) \
	                 --test_size $(TEST_SIZE) \
					 --random_state $(RANDOM_STATE)

$(MODELS): $(SRC_PATH)/train_models.py $(INTERIM_PATH)/bank_train.csv
	$(POETRY_RUN) $< $(word 2, $^) \
	                 $(notdir $(basename $@)) \
					 $@ \
					 --cv $(CV) \
					 --scoring $(SCORING) \
					 --n_iter $(N_ITER) \
					 --random_state $(RANDOM_STATE)

clean:
	rm $(MODELS)