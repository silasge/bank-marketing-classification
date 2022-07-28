# Data URL
BANK_MARKETING_URL = https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip

# Diretórios 
EXTERNAL_DATA_PATH = ./data/external
RAW_DATA_PATH = ./data/raw
INTERIM_PATH = ./data/interim
REFERENCES_PATH = ./references
MODELS_PATH = ./models

# Parâmetros
TEST_SIZE = 0.25
RANDOM_STATE = 42
CV = 5
N_ITER = 20
SCORING = roc_auc

# Modelos

MODELS = $(MODELS_PATH)/lr.pkl $(MODELS_PATH)/svc.pkl $(MODELS_PATH)/dt.pkl $(MODELS_PATH)/rf.pkl


all: $(MODELS)
	
$(EXTERNAL_DATA_PATH)/bank-additional.zip:
	@echo ">>> Baixando bank-additional.zip em $(EXTERNAL_DATA_PATH)"
	poetry run download_data $(BANK_MARKETING_URL) $(EXTERNAL_DATA_PATH)


$(RAW_DATA_PATH)/bank-additional-full.csv: $(EXTERNAL_DATA_PATH)/bank-additional.zip
	@echo ">>> Extraindo bank-additional-full.csv em $(RAW_DATA_PATH)..."
	poetry run unzip_data $< bank-additional/bank-additional-full.csv $(RAW_DATA_PATH)

$(REFERENCES_PATH)/bank-additional-names.txt: $(EXTERNAL_DATA_PATH)/bank-additional.zip
	@echo ">>> Extraindo bank-additional-names.txt em $(REFERENCES_PATH)..."
	poetry run unzip_data $< bank-additional/bank-additional-names.txt $(REFERENCES_PATH)

train_test_split: $(RAW_DATA_PATH)/bank-additional-full.csv
	@echo ">>> Dividindo conjunto de treinamento e teste em $(INTERIM_PATH)"
	poetry run split_data $< \
	                      $(INTERIM_PATH) \
	                      --test_size $(TEST_SIZE) \
						  --random_state $(RANDOM_STATE)

$(MODELS): train_test_split
	@echo Treinando modelo $(notdir $(basename $@))
	poetry run train_models $(INTERIM_PATH)/bank_train.csv \
	                        $(notdir $(basename $@)) \
							$@ \
							--cv $(CV) \
							--scoring $(SCORING) \
							--n_iter $(N_ITER) \
							--random_state $(RANDOM_STATE)

clean:
	rm $(MODELS)