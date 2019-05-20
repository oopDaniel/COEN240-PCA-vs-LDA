remove:
	rm -rf ./tmp

install:
	@echo Creating virtual environment...
	python3 -m venv tmp
	@echo Installing pkg...
	./tmp/bin/pip3 install numpy matplotlib sklearn opencv-python

# modify the arguments here: folder_path / experiments / d0
start:
	./tmp/bin/python3 main.py att_faces_10 20 40

all:
	make remove install start

.PHONY: remove install start all