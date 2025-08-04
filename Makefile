venv:
	uv venv

download-data:
	export KAGGLE_USERNAME=<KAGGLE_USERNAME>
	export KAGGLE_KEY=<KAGGLE_KEY>
	mkdir -p data
	kaggle competitions download -c playground-series-s5e8 -p data
	unzip data/playground-series-s5e8.zip -d data
	rm data/playground-series-s5e8.zip
