conda activate bachelor-project-env
zip './src' './kaggle_src/src.zip' -Force
kaggle datasets version -p './kaggle_src' -m 'update src'