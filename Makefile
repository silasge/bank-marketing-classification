BANK_MARKETING_URL = https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
EXTERNAL_DATA_PATH = ./data/external
RAW_DATA_PATH = ./data/raw
INTERIM_PATH = ./data/interim
REFERENCES_PATH = ./references
MODELS_PATH = ./models
SRC_PATH = ./bank_marketing_classification
N_ITER = 20
CV = 5
TEST_SIZE = 0.25

.PHONY: virtual_env

all: $(MODELS_PATH)/lr.pkl \
$(MODELS_PATH)/svc.pkl \
$(MODELS_PATH)/dt.pkl \
$(MODELS_PATH)/rf.pkl \
$(MODELS_PATH)/gb.pkl

virtual_env: 
	@echo ">>> Ativando o ambiente virtual..."
	. $(.venv/Scripts)/activate.ps1

$(EXTERNAL_DATA_PATH)/bank-additional.zip:
	@echo ">>> Baixando bank-additional.zip em $(EXTERNAL_DATA_PATH)"
	python $(SRC_PATH)/data/download_data.py --url $(BANK_MARKETING_URL) \
	                                         --to_path $(EXTERNAL_DATA_PATH)

$(RAW_DATA_PATH)/bank-additional-full.csv: $(EXTERNAL_DATA_PATH)/bank-additional.zip
	@echo ">>> Extraindo bank-additional-full.csv em $(RAW_DATA_PATH)..."
	python $(SRC_PATH)/data/unzip_data.py --zip_file $< \
	                                      --member bank-additional/bank-additional-full.csv \
										  --to_path $(RAW_DATA_PATH)

$(REFERENCES_PATH)/bank-additional-names.txt: $(EXTERNAL_DATA_PATH)/bank-additional.zip
	@echo ">>> Extraindo bank-additional-names.txt em $(REFERENCES_PATH)..."
	python $(SRC_PATH)/data/unzip_data.py --zip_file $< \
	                                      --member bank-additional/bank-additional-names.txt \
										  --to_path $(REFERENCES_PATH)

$(INTERIM_PATH)/bank_train.csv \
$(INTERIM_PATH)/bank_teste.csv: $(RAW_DATA_PATH)/bank-additional-full.csv
	@echo ">>> Salvando conjunto de treinamento..."
	python $(SRC_PATH)/data/split_data.py --csv_file $< 
										  --test_size $(TEST_SIZE)
										  --save_to $@

$(MODELS_PATH)/lr.pkl \
$(MODELS_PATH)/svc.pkl \
$(MODELS_PATH)/dt.pkl \
$(MODELS_PATH)/rf.pkl \
$(MODELS_PATH)/gb.pkl: $(INTERIM_PATH)/bank_train.csv
	@echo Treinando modelo $(notdir $(basename $@))
	python $(SRC_PATH)/models/train_models.py --train_data $< \
	                                          --model $(notdir $(basename $@)) \
											  --save_to $@ \
											  --cv $(CV) \
											  --n_iter $(N_ITER)





