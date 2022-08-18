all: poetry_install_env references predict

.PHONY: clean_downloads clean_splits clean_models clean_altair black predict

# Data URL
BANK_MARKETING_URL = https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip

# Diretórios 
EXTERNAL_DATA_PATH = ./data/external
RAW_DATA_PATH = ./data/raw
SPLITS_DATA_PATH = ./data/splits
PREDICTIONS_DATA_PATH = ./data/predictions
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
THRESHOLD = 0.4

# Comandos que dependem do OS
ifeq ($(OS),Windows_NT)
	RM = del /Q
	FixPathIfWin = $(subst /,\,$1)
else
	RM = rm -f
	FixPathIfWin = $1
endif

poetry_install_env:
	@poetry install

download: $(EXTERNAL_DATA_PATH)/bank-additional.zip
unzip: $(RAW_DATA_PATH)/bank-additional-full.csv
references: $(REFERENCES_PATH)/bank-additional-names.txt
split: $(SPLITS_DATA_PATH)/bank_train.csv $(SPLITS_DATA_PATH)/bank_test.csv
models: $(MODELS)
predict: $(PREDICTIONS_DATA_PATH)/bank_test_predictions.csv

# Modelos
MODELS = $(MODELS_PATH)/lr.pkl $(MODELS_PATH)/svc.pkl $(MODELS_PATH)/dt.pkl $(MODELS_PATH)/rf.pkl

# Targets
$(EXTERNAL_DATA_PATH)/bank-additional.zip: $(SRC_PATH)/download_data.py
	@$(POETRY_RUN) $< $(BANK_MARKETING_URL) $(EXTERNAL_DATA_PATH)

$(RAW_DATA_PATH)/bank-additional-full.csv: $(SRC_PATH)/unzip_data.py $(EXTERNAL_DATA_PATH)/bank-additional.zip
	@$(POETRY_RUN) $< $(word 2, $^) bank-additional/bank-additional-full.csv $(RAW_DATA_PATH)

$(REFERENCES_PATH)/bank-additional-names.txt: $(SRC_PATH)/unzip_data.py $(EXTERNAL_DATA_PATH)/bank-additional.zip
	@$(POETRY_RUN) $< $(word 2, $^) bank-additional/bank-additional-names.txt $(REFERENCES_PATH)

$(SPLITS_DATA_PATH)/bank_train.csv $(SPLITS_DATA_PATH)/bank_test.csv &: $(SRC_PATH)/split_data.py $(RAW_DATA_PATH)/bank-additional-full.csv
	@$(POETRY_RUN) $< $(word 2, $^) \
	                 $(SPLITS_DATA_PATH) \
	                 --test_size $(TEST_SIZE) \
					 --random_state $(RANDOM_STATE)

$(MODELS): $(SRC_PATH)/train_models.py $(SPLITS_DATA_PATH)/bank_train.csv
	@$(POETRY_RUN) $< $(word 2, $^) \
	                 $(notdir $(basename $@)) \
					 $@ \
					 --cv $(CV) \
					 --scoring roc_auc \
					 --n_iter $(N_ITER) \
					 --random_state $(RANDOM_STATE)

$(PREDICTIONS_DATA_PATH)/bank_test_predictions.csv: $(SRC_PATH)/make_predictions.py $(SPLITS_DATA_PATH)/bank_test.csv $(MODELS)
	@$(POETRY_RUN) $< --test_set $(word 2, $^) \
	                 --threshold $(THRESHOLD) \
	                 --models $(MODELS) \
					 --save_to $(dir $@)

clean_downloads:
	@$(RM) $(call FixPathIfWin, $(EXTERNAL_DATA_PATH)/*.zip)
	@$(RM) $(call FixPathIfWin, $(RAW_DATA_PATH)/*.csv)
	@$(RM) $(call FixPathIfWin, $(REFERENCES_PATH)/*.txt)
	@echo Downloads excluidos.

clean_splits:
	@$(RM) $(call FixPathIfWin, $(SPLITS_DATA_PATH)/*.csv)
	@echo Splits excluidos.

clean_models:
	@$(RM) $(call FixPathIfWin, $(MODELS))
	@echo Modelos excluidos.

clean_predictions:
	@$(RM) $(call FixPathIfWin, $(PREDICTIONS_DATA_PATH)/*.csv)
	@echo Previsões excluidas.

clean_altair:
	@$(RM) $(call FixPathIfWin, notebooks/*.json)
	@echo Arquivos do altair excluidos.

black:
	poetry run black ./bank_marketing_classification
